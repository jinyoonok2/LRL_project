"""Helpers for relating articulated door panels to physical doorways."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DoorPanelInfo:
    """Scene-metadata description of one articulated door panel."""

    doorway_name: str
    asset_id: str | None
    panel_count: int

    @property
    def is_single_panel(self) -> bool:
        return self.panel_count == 1


def door_panel_info_by_body_name(
    scene_metadata: dict | None,
) -> dict[str, DoorPanelInfo]:
    """Map each door-panel body name to its containing physical doorway."""

    panel_info: dict[str, DoorPanelInfo] = {}
    for doorway_name, metadata in (scene_metadata or {}).get("objects", {}).items():
        if "doorway" not in doorway_name.lower():
            continue

        body_name_map = metadata.get("name_map", {}).get("bodies", {})
        panel_body_names = [
            body_name
            for body_name, source_body_name in body_name_map.items()
            if "_door_" in source_body_name.lower()
        ]
        info = DoorPanelInfo(
            doorway_name=doorway_name,
            asset_id=metadata.get("asset_id"),
            panel_count=len(panel_body_names),
        )
        for panel_body_name in panel_body_names:
            panel_info[panel_body_name] = info

    return panel_info


def filter_single_panel_door_names(
    door_body_names: list[str],
    scene_metadata: dict | None,
) -> tuple[list[str], list[str]]:
    """Partition candidates into confirmed single-panel and rejected doors.

    Unclassified candidates are rejected in strict mode instead of being
    accidentally admitted as single-panel doors.
    """

    panel_info = door_panel_info_by_body_name(scene_metadata)
    single_panel = [
        name
        for name in door_body_names
        if name in panel_info and panel_info[name].is_single_panel
    ]
    rejected = [name for name in door_body_names if name not in single_panel]
    return single_panel, rejected
