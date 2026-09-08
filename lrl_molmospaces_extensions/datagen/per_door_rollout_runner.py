"""Per-door rollout runner layered on the unmodified MolmoSpaces pipeline."""

from pathlib import Path

from molmo_spaces.data_generation.pipeline import *  # noqa: F403
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner


class PerDoorRolloutRunner(ParallelRolloutRunner):
    """Rollout runner with per-door completion and durable trial accounting."""

    @staticmethod
    def _sampler_complete(task_sampler, house_id: int) -> bool:
        completion_check = getattr(task_sampler, "is_house_sampling_complete", None)
        return completion_check is not None and bool(completion_check(house_id))

    @staticmethod
    def load_episodes_for_house(
        exp_config,
        house_id,
        batch_suffix,
        worker_task_sampler,
        worker_logger,
    ):
        if getattr(exp_config.task_sampler_config, "enumerate_all_doors", False):
            return (
                [None] * exp_config.task_sampler_config.max_house_episode_attempts,
                worker_task_sampler,
            )
        return ParallelRolloutRunner.load_episodes_for_house(
            exp_config,
            house_id,
            batch_suffix,
            worker_task_sampler,
            worker_logger,
        )

    @staticmethod
    def _report_outcome(
        task_sampler,
        task,
        success: bool,
        *,
        seed: int | None,
        termination_reason: str,
        exception: BaseException | None = None,
    ) -> Path | None:
        report = getattr(task_sampler, "report_rollout_result", None)
        if report is None:
            return None
        return report(
            task,
            success,
            seed=seed,
            termination_reason=termination_reason,
            exception=exception,
        )

    @staticmethod
    def _checkpoint_episode(
        worker_logger,
        episode_info: dict | None,
        house_output_dir: Path,
        exp_config,
        batch_suffix: str,
        checkpoint_index: int,
        task_sampler,
        record_path: Path | None,
        datagen_profiler,
        batch_num,
        total_batches,
    ) -> None:
        h5_path = house_output_dir / (
            f"trajectories{batch_suffix}_checkpoint_{checkpoint_index:06d}.h5"
        )
        if episode_info is not None:
            save_house_trajectories(
                worker_logger,
                [episode_info],
                house_output_dir,
                exp_config,
                f"{batch_suffix}_checkpoint_{checkpoint_index:06d}",
                datagen_profiler,
                batch_num,
                total_batches,
            )
        update_status = getattr(task_sampler, "update_trial_h5_status", None)
        if update_status is not None:
            update_status(
                record_path,
                h5_saved=h5_path.exists(),
                h5_path=h5_path if h5_path.exists() else None,
            )

    @staticmethod
    def process_single_house(
        worker_id: int,
        worker_logger,
        house_id: int,
        exp_config: MlSpacesExpConfig,
        samples_per_house: int,
        shutdown_event,
        task_sampler,
        preloaded_policy: BasePolicy | None = None,
        max_allowed_sequential_task_sampler_failures: int = 10,
        max_allowed_sequential_rollout_failures: int = 10,
        filter_for_successful_trajectories: bool = False,
        runner_class=None,
        batch_num: int | None = None,
        total_batches: int | None = None,
        datagen_profiler: DatagenProfiler | None = None,
    ) -> tuple[int, int, bool]:
        """
        Process all episodes for a single house using customizable hooks.

        This method uses a while loop to iterate over episodes, calling hook methods
        via runner_class to allow subclasses to customize behavior without duplicating
        the entire method.

        Hooks called (override in subclass to customize):
        - load_episodes_for_house: Load episode specs from source (JSON, etc.)
        - get_max_episode_attempts: Maximum iterations of the episode loop
        - should_stop_early: Whether to stop before max attempts (e.g., enough successes)
        - prepare_episode_config: Modify config per-episode
        - get_episode_task_sampler: Get/create task sampler for episode
        - sample_task_from_spec: Sample task from specification
        - get_episode_seed: Get seed for episode
        - should_close_episode_task_sampler: Whether to close sampler per-episode

        Args:
            worker_id: ID of the worker thread/process
            worker_logger: Logger instance for this worker
            house_id: Index of the house to process
            exp_config: Experiment configuration
            samples_per_house: Number of episodes to collect for this house
            shutdown_event: Event to signal shutdown
            task_sampler: Task sampler instance (shared across houses for this worker)
            preloaded_policy: Optional pre-initialized policy instance
            max_allowed_sequential_task_sampler_failures: Max consecutive task sampling failures
            max_allowed_sequential_rollout_failures: Max consecutive rollout failures
            filter_for_successful_trajectories: Whether to filter for successful trajectories only
            runner_class: Runner class with hook methods to call
            batch_num: Batch number for this house (for batched processing)
            total_batches: Total number of batches for this house
            datagen_profiler: DatagenProfiler for per-worker timing (optional)

        Returns:
            tuple: (house_success_count, house_total_count, irrecoverable_failure_flag)
        """
        house_success_count = 0
        house_total_count = 0
        irrecoverable_failure_in_house = False

        # Setup directories and check for existing output
        house_output_dir, house_debug_dir, batch_suffix, should_skip = setup_house_dirs(
            exp_config, house_id, batch_num, total_batches
        )
        if should_skip:
            worker_logger.info(
                f"SKIPPING HOUSE {house_id} BATCH {batch_num}/{total_batches}: "
                f"Output already exists at {house_output_dir / f'trajectories{batch_suffix}.h5'}"
            )
            return 0, 0, False

        # Load episodes using hook - allows subclasses to load from different scene datasets
        episode_specs, shared_task_sampler = runner_class.load_episodes_for_house(
            exp_config, house_id, batch_suffix, task_sampler, worker_logger
        )

        if not episode_specs:
            worker_logger.warning(f"No episodes to process for house {house_id}")
            return 0, 0, False

        max_attempts = runner_class.get_max_episode_attempts(
            episode_specs, samples_per_house, exp_config
        )

        # Collect raw history data for this house
        house_raw_histories = []
        house_debug_raw_histories = []

        # Sequential failure tracking
        num_sequential_task_sampler_failures = 0
        num_sequential_rollout_failures = 0
        viewer = None
        checkpoint_index = 0

        # While loop with explicit index - allows subclasses to customize iteration
        episode_idx = 0
        while episode_idx < max_attempts:
            # Check early stop condition (e.g., enough successes for datagen)
            if getattr(exp_config.task_sampler_config, "enumerate_all_doors", False):
                should_stop = runner_class._sampler_complete(task_sampler, house_id)
            else:
                should_stop = runner_class.should_stop_early(
                    len(house_raw_histories),
                    samples_per_house,
                    exp_config=exp_config,
                )
            if should_stop:
                break

            # Check for shutdown signal
            if shutdown_event.is_set():
                worker_logger.info(f"Worker {worker_id} house {house_id} received shutdown signal")
                irrecoverable_failure_in_house = True
                break

            # Check for too many consecutive task sampling failures
            if num_sequential_task_sampler_failures >= max_allowed_sequential_task_sampler_failures:
                worker_logger.error(
                    f"Worker {worker_id} house {house_id} encountered "
                    f"{num_sequential_task_sampler_failures} consecutive task sampling failures. "
                    "This is unrecoverable."
                )
                irrecoverable_failure_in_house = True
                break

            # Check for too many consecutive rollout failures
            if num_sequential_rollout_failures >= max_allowed_sequential_rollout_failures:
                worker_logger.error(
                    f"Worker {worker_id} house {house_id} rollout failed across "
                    f"{num_sequential_rollout_failures} retries. This is irrecoverable."
                )
                irrecoverable_failure_in_house = True
                break

            # Get episode spec for this iteration
            episode_spec = runner_class.get_episode_spec_at_index(episode_specs, episode_idx)

            # Track state for this episode
            task = None
            policy = None
            episode_task_sampler = None
            success = False
            task_sampling_failed = False
            house_invalid = False
            house_sampling_complete = False
            episode_seed = None
            object_name = "unknown"
            rollout_started = False

            if datagen_profiler is not None:
                datagen_profiler.start("episode_total")

            # Prepare episode-specific config
            episode_config = runner_class.prepare_episode_config(
                exp_config, episode_spec, episode_idx
            )

            with cleanup_context():
                if viewer is not None:
                    viewer.close()
                    viewer = None

                # Task sampling phase
                task_sampling_start = time.perf_counter()

                try:
                    # Get task sampler for this episode (shared or per-episode)
                    episode_task_sampler = runner_class.get_episode_task_sampler(
                        episode_config, episode_spec, shared_task_sampler, datagen_profiler
                    )

                    # Sample task
                    task = runner_class.sample_task_from_spec(
                        episode_task_sampler, house_id, episode_spec, episode_idx
                    )

                    if task is None:
                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx}: "
                            "task sampling returned None"
                        )
                        house_invalid = True
                    else:
                        # Record successful sampling time
                        if datagen_profiler is not None:
                            datagen_profiler.record(
                                "task_sampling", time.perf_counter() - task_sampling_start
                            )
                            task.set_datagen_profiler(datagen_profiler)

                        num_sequential_task_sampler_failures = 0

                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx}/{max_attempts} "
                            f"collected={len(house_raw_histories)}/{samples_per_house}"
                        )

                except HouseInvalidForTask as e:
                    if runner_class._sampler_complete(
                        episode_task_sampler, house_id
                    ):
                        worker_logger.info(
                            f"Worker {worker_id} house {house_id}: "
                            "all per-door outcomes have been exhausted"
                        )
                        house_sampling_complete = True
                    else:
                        traceback.print_exc()
                        worker_logger.warning(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} "
                            f"HouseInvalidForTask: {e.reason}"
                        )
                        house_invalid = True
                    if datagen_profiler is not None:
                        datagen_profiler.record(
                            "task_sampling_failed", time.perf_counter() - task_sampling_start
                        )

                except Exception as e:
                    traceback.print_exc()
                    worker_logger.error(
                        f"Worker {worker_id} house {house_id} episode {episode_idx} "
                        f"task sampling error: {str(e)}"
                    )
                    num_sequential_task_sampler_failures += 1
                    task_sampling_failed = True
                    if datagen_profiler is not None:
                        datagen_profiler.record(
                            "task_sampling_failed", time.perf_counter() - task_sampling_start
                        )

                # Rollout phase (only if task sampling succeeded)
                if task is not None and not house_invalid and not task_sampling_failed:
                    try:
                        # Setup policy and viewer
                        policy = setup_policy(
                            episode_config, task, preloaded_policy, datagen_profiler
                        )
                        viewer = setup_viewer(episode_config, task, policy, viewer)

                        # Get episode seed
                        episode_seed = runner_class.get_episode_seed(
                            episode_idx, episode_spec, episode_task_sampler
                        )

                        # Run the rollout
                        rollout_started = True
                        success = runner_class.run_single_rollout(
                            episode_seed=episode_seed,
                            task=task,
                            policy=policy,
                            profiler=episode_config.profiler,
                            viewer=viewer,
                            shutdown_event=shutdown_event,
                            datagen_profiler=datagen_profiler,
                            end_on_success=exp_config.end_on_success,
                        )

                        num_sequential_rollout_failures = 0

                        # Extract object name for logging if available
                        object_name = "unknown"
                        if hasattr(task, "config") and hasattr(task.config, "task_config"):
                            if hasattr(task.config.task_config, "pickup_obj_name"):
                                object_name = task.config.task_config.pickup_obj_name

                        worker_logger.info(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} "
                            f"object {object_name} completed with success={success}"
                        )

                        # Collect trajectory
                        should_save = success or not filter_for_successful_trajectories
                        history = task.get_history()
                        should_save_debug = not should_save and random.random() < 0.01
                        episode_info = None

                        if should_save or should_save_debug:
                            episode_info = {
                                "history": history,
                                "sensor_suite": task.sensor_suite,
                                "success": success,
                                "seed": episode_seed,
                            }
                            if should_save_debug:
                                house_debug_raw_histories.append(episode_info)
                                episode_info = None
                                worker_logger.info(
                                    f"Queueing failed trajectory for debug (seed: {episode_seed})"
                                )
                        else:
                            del history

                        # Update house counters
                        house_total_count += 1
                        if success:
                            house_success_count += 1
                        else:
                            # Report failure for this asset (may lead to dynamic blacklisting)
                            asset_uid = task_sampler.get_asset_uid_from_object(
                                task.env, object_name
                            )
                            if asset_uid:
                                task_sampler.report_asset_failure(
                                    asset_uid, "rollout failed (e.g., IK failure)"
                                )

                        record_path = runner_class._report_outcome(
                            episode_task_sampler,
                            task,
                            success,
                            seed=episode_seed,
                            termination_reason=(
                                "success"
                                if success
                                else "policy_done_or_horizon"
                            ),
                        )
                        checkpoint_index += 1
                        runner_class._checkpoint_episode(
                            worker_logger,
                            episode_info,
                            house_output_dir,
                            exp_config,
                            batch_suffix,
                            checkpoint_index,
                            episode_task_sampler,
                            record_path,
                            datagen_profiler,
                            batch_num,
                            total_batches,
                        )

                        if datagen_profiler is not None:
                            datagen_profiler.end("episode_total")
                            datagen_profiler.log_episode_summary(
                                episode_idx=episode_idx,
                                house_id=house_id,
                                success=success,
                            )

                    except Exception as e:
                        worker_logger.error(
                            f"Worker {worker_id} house {house_id} episode {episode_idx} rollout error: {str(e)}"
                        )
                        traceback.print_exc()
                        if not rollout_started:
                            num_sequential_rollout_failures += 1
                            if datagen_profiler is not None:
                                datagen_profiler.end("episode_total")
                            cleanup_episode_resources(
                                task=task,
                                policy=policy,
                                task_sampler=episode_task_sampler,
                                preloaded_policy=preloaded_policy,
                                close_task_sampler=(
                                    runner_class.should_close_episode_task_sampler()
                                ),
                            )
                            episode_idx += 1
                            continue
                        # A planner exception is an executed failed trial, not a
                        # sampling failure. Count it and continue the fixed trial set.
                        num_sequential_rollout_failures = 0
                        house_total_count += 1
                        episode_info = None
                        try:
                            history = task.get_history()
                            observations = history.get("observations", [])
                            if observations and not filter_for_successful_trajectories:
                                episode_info = {
                                    "history": history,
                                    "sensor_suite": task.sensor_suite,
                                    "success": False,
                                    "seed": episode_seed,
                                }
                        except Exception as history_error:
                            worker_logger.warning(
                                "Could not collect partial history after planner failure: %s",
                                history_error,
                            )

                        record_path = runner_class._report_outcome(
                            episode_task_sampler,
                            task,
                            False,
                            seed=episode_seed,
                            termination_reason="planner_exception",
                            exception=e,
                        )
                        checkpoint_index += 1
                        runner_class._checkpoint_episode(
                            worker_logger,
                            episode_info,
                            house_output_dir,
                            exp_config,
                            batch_suffix,
                            checkpoint_index,
                            episode_task_sampler,
                            record_path,
                            datagen_profiler,
                            batch_num,
                            total_batches,
                        )

                        # Report failure for this asset (may lead to dynamic blacklisting)
                        try:
                            asset_uid = task_sampler.get_asset_uid_from_object(
                                task.env, object_name
                            )
                            if asset_uid:
                                task_sampler.report_asset_failure(
                                    asset_uid, f"rollout exception: {e}"
                                )
                        except Exception:
                            pass  # Don't let failure tracking break the error handling

                        if datagen_profiler is not None:
                            datagen_profiler.end("episode_total")

                else:
                    # Task sampling failed or house invalid
                    if datagen_profiler is not None:
                        datagen_profiler.end("episode_total")

                # Cleanup resources
                cleanup_episode_resources(
                    task=task,
                    policy=policy,
                    task_sampler=episode_task_sampler,
                    preloaded_policy=preloaded_policy,
                    close_task_sampler=runner_class.should_close_episode_task_sampler(),
                )

            # Handle normal per-door completion or a genuinely invalid house.
            if house_sampling_complete:
                break
            if house_invalid:
                irrecoverable_failure_in_house = True
                break

            # Always increment episode index
            episode_idx += 1

        # Cleanup viewer
        if viewer is not None:
            viewer.close()
            viewer = None

        # Check shutdown signal before saving
        if shutdown_event.is_set():
            worker_logger.info(
                f"Worker {worker_id} house {house_id} shutdown requested, skipping save"
            )
            return house_success_count, house_total_count, True

        # Per-door trajectories are checkpointed immediately after each trial.
        if house_raw_histories:
            save_house_trajectories(
                worker_logger,
                house_raw_histories,
                house_output_dir,
                exp_config,
                batch_suffix,
                datagen_profiler,
                batch_num,
                total_batches,
            )

        # Save debug trajectories
        if house_debug_raw_histories:
            save_house_trajectories(
                worker_logger,
                house_debug_raw_histories,
                house_debug_dir,
                exp_config,
                batch_suffix,
                datagen_profiler=None,
                batch_num=batch_num,
                total_batches=total_batches,
            )

        worker_logger.info(
            f"Worker {worker_id} completed house {house_id}: "
            f"{house_success_count}/{house_total_count} successful episodes"
        )

        if datagen_profiler is not None:
            datagen_profiler.log_house_summary(
                house_id=house_id,
                success_count=house_success_count,
                total_count=house_total_count,
            )

        return house_success_count, house_total_count, irrecoverable_failure_in_house

