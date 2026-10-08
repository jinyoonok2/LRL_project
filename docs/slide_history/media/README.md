# Full validation comparison videos

All 90 paired MP4s: expert ground truth on the left, model inference on the right, 10 fps at 1x playback. The shorter clip holds its final frame. GIFs have been removed.

[Opened with qualifying grasp (1)](success/opened_with_grasp/README.md)

[Opened without qualifying grasp (27)](success/opened_without_grasp/README.md)

[Did not open successfully (62)](failure/README.md)

The existing dataset filter requires at least 30 cumulative recorded grasp frames (about 3 seconds at 10 Hz). Below-threshold does not necessarily mean zero grasp contact. The simulator stops successful rollouts upon task success, which can limit recorded grasp duration. Opening success is 28/90 (31.1%); success with qualifying grasp is 1/90 (1.1%).

Metrics: [grasp index](grasp_index.tsv), [summary](grasp_summary.json), and [source/video manifest](manifest.json).
