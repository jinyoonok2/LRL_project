"""Generate PowerPoint-style RBY1 MolmoSpaces/MolmoBot diagrams as PNG files.

This temporary presentation helper renders a compact three-diagram set:

1. MolmoSpaces benchmark/simulation data flow
2. RBY1 benchmark and episode overview
3. MolmoSpaces planner data generation -> MolmoBot learned policy flow

Usage:
    conda run -n mlspaces python render_rby1_ascii_diagrams.py
"""

from __future__ import annotations

import math
import textwrap
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "rby1_diagram_images"


def load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def load_bold_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return load_font(size)


FONT_TITLE = load_bold_font(58)
FONT_SUBTITLE = load_font(32)
FONT_SECTION = load_bold_font(34)
FONT_BOX_TITLE = load_bold_font(31)
FONT_BODY = load_font(29)
FONT_SMALL = load_font(26)

BG = (248, 250, 252)
TEXT = (24, 31, 42)
MUTED = (77, 91, 113)
LINE = (74, 85, 104)
BLUE = (226, 241, 255)
BLUE_DARK = (38, 95, 156)
GREEN = (230, 247, 237)
GREEN_DARK = (38, 125, 74)
ORANGE = (255, 242, 223)
ORANGE_DARK = (170, 101, 22)
PURPLE = (242, 234, 255)
PURPLE_DARK = (111, 70, 166)
GRAY = (236, 240, 245)
GRAY_DARK = (83, 95, 113)


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    title: str
    lines: tuple[str, ...]
    fill: tuple[int, int, int] = BLUE
    outline: tuple[int, int, int] = BLUE_DARK


def text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def wrapped_lines(text: str, max_chars: int) -> list[str]:
    return textwrap.wrap(text, width=max_chars, break_long_words=False) or [text]


def draw_title(draw: ImageDraw.ImageDraw, title: str, subtitle: str | None = None) -> None:
    draw.text((70, 38), title, font=FONT_TITLE, fill=TEXT)
    if subtitle:
        draw.text((72, 104), subtitle, font=FONT_SUBTITLE, fill=MUTED)


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill: tuple[int, int, int] = TEXT,
) -> None:
    x1, y1, x2, y2 = box
    w, h = text_size(draw, text, font)
    draw.text((x1 + (x2 - x1 - w) / 2, y1 + (y2 - y1 - h) / 2), text, font=font, fill=fill)


def draw_box(draw: ImageDraw.ImageDraw, box: Box, radius: int = 24) -> None:
    xy = (box.x, box.y, box.x + box.w, box.y + box.h)
    draw.rounded_rectangle(xy, radius=radius, fill=box.fill, outline=box.outline, width=3)
    draw.text((box.x + 22, box.y + 18), box.title, font=FONT_BOX_TITLE, fill=box.outline)

    y = box.y + 72
    max_chars = max(16, int(box.w / 16))
    for raw_line in box.lines:
        for line in wrapped_lines(raw_line, max_chars):
            draw.text((box.x + 24, y), line, font=FONT_BODY, fill=TEXT)
            y += 43
        y += 3


def edge_point(box: Box, side: str) -> tuple[int, int]:
    if side == "left":
        return box.x, box.y + box.h // 2
    if side == "right":
        return box.x + box.w, box.y + box.h // 2
    if side == "top":
        return box.x + box.w // 2, box.y
    if side == "bottom":
        return box.x + box.w // 2, box.y + box.h
    raise ValueError(side)


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[int, int],
    end: tuple[int, int],
    color: tuple[int, int, int] = LINE,
    width: int = 4,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_len = 18
    arrow_angle = math.radians(28)
    p1 = (
        end[0] - arrow_len * math.cos(angle - arrow_angle),
        end[1] - arrow_len * math.sin(angle - arrow_angle),
    )
    p2 = (
        end[0] - arrow_len * math.cos(angle + arrow_angle),
        end[1] - arrow_len * math.sin(angle + arrow_angle),
    )
    draw.polygon([end, p1, p2], fill=color)


def draw_polyline_arrow(
    draw: ImageDraw.ImageDraw,
    points: list[tuple[int, int]],
    color: tuple[int, int, int] = LINE,
    width: int = 4,
) -> None:
    if len(points) < 2:
        raise ValueError("A polyline arrow needs at least two points.")
    draw.line(points, fill=color, width=width, joint="curve")
    start, end = points[-2], points[-1]
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    arrow_len = 18
    arrow_angle = math.radians(28)
    p1 = (
        end[0] - arrow_len * math.cos(angle - arrow_angle),
        end[1] - arrow_len * math.sin(angle - arrow_angle),
    )
    p2 = (
        end[0] - arrow_len * math.cos(angle + arrow_angle),
        end[1] - arrow_len * math.sin(angle + arrow_angle),
    )
    draw.polygon([end, p1, p2], fill=color)


def new_canvas(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (width, height), BG)
    return image, ImageDraw.Draw(image)


def save(image: Image.Image, name: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image.save(OUTPUT_DIR / name)
    print(OUTPUT_DIR / name)


def draw_table(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    col_widths: list[int],
    row_heights: list[int],
    headers: list[str],
    rows: list[list[str]],
) -> None:
    total_w = sum(col_widths)
    total_h = sum(row_heights)
    draw.rounded_rectangle(
        (x, y, x + total_w, y + total_h),
        radius=24,
        fill=(255, 255, 255),
        outline=LINE,
        width=3,
    )

    header_h = row_heights[0]
    draw.rounded_rectangle(
        (x, y, x + total_w, y + header_h),
        radius=24,
        fill=GRAY,
        outline=LINE,
        width=0,
    )
    draw.rectangle((x, y + header_h - 24, x + total_w, y + header_h), fill=GRAY)

    cur_x = x
    for idx, header in enumerate(headers):
        draw.text((cur_x + 16, y + 20), header, font=FONT_BOX_TITLE, fill=GRAY_DARK)
        cur_x += col_widths[idx]

    cur_y = y + header_h
    for row_idx, row in enumerate(rows):
        row_h = row_heights[row_idx + 1]
        fill = (255, 255, 255) if row_idx % 2 == 0 else (245, 248, 252)
        draw.rectangle((x, cur_y, x + total_w, cur_y + row_h), fill=fill)
        cur_x = x
        for col_idx, value in enumerate(row):
            max_chars = max(10, int(col_widths[col_idx] / 14.5))
            text_y = cur_y + 14
            for line in wrapped_lines(value, max_chars):
                draw.text((cur_x + 16, text_y), line, font=FONT_SMALL, fill=TEXT)
                text_y += 36
            cur_x += col_widths[col_idx]
        cur_y += row_h

    cur_x = x
    for width in col_widths[:-1]:
        cur_x += width
        draw.line((cur_x, y, cur_x, y + total_h), fill=(210, 218, 229), width=2)
    cur_y = y
    for height in row_heights[:-1]:
        cur_y += height
        draw.line((x, cur_y, x + total_w, cur_y), fill=(210, 218, 229), width=2)


def render_molmospaces_benchmark_flow() -> None:
    image, draw = new_canvas(2300, 1450)
    draw_title(
        draw,
        "MolmoSpaces Benchmark Flow",
        "Prepare these data/assets to recreate fixed RBY1 simulation episodes and save H5 + MP4 rollouts.",
    )

    input_boxes = [
        Box(80, 185, 380, 220, "Scene / World", ("dataset + split", "house index", "scene XML / ID"), BLUE, BLUE_DARK),
        Box(500, 185, 380, 220, "Object Assets", ("mesh / XML files", "Objaverse object IDs", "materials/textures"), GREEN, GREEN_DARK),
        Box(920, 185, 380, 220, "Object Poses", ("object positions", "object orientations", "target start/goal pose"), ORANGE, ORANGE_DARK),
        Box(1340, 185, 380, 220, "Robot State", ("RBY1 robot name", "initial qpos", "base pose / gripper"), GREEN, GREEN_DARK),
        Box(1760, 185, 420, 220, "Cameras + Language", ("camera names/poses", "resolution/intrinsics", "instruction text"), PURPLE, PURPLE_DARK),
    ]
    optional = Box(
        80,
        515,
        500,
        285,
        "Optional Grasp Assets",
        (
            "Used by rigid pick / pick-and-place planners",
            "Not used directly by MolmoBot at runtime",
        ),
        ORANGE,
        ORANGE_DARK,
    )
    benchmark = Box(
        770,
        520,
        520,
        225,
        "benchmark.json",
        ("fixed EpisodeSpec list", "one row per reproducible task", "select with benchmark idx"),
        PURPLE,
        PURPLE_DARK,
    )
    runner = Box(
        1480,
        520,
        520,
        225,
        "JsonEvalRunner",
        ("loads selected episode", "builds JsonEvalTaskSampler", "reconstructs scene/task"),
        BLUE,
        BLUE_DARK,
    )
    sim = Box(
        520,
        955,
        520,
        245,
        "MolmoSpaces + MuJoCo",
        ("loads scene and robot", "places objects/cameras", "runs planner or learned policy"),
        GREEN,
        GREEN_DARK,
    )
    output = Box(
        1260,
        955,
        520,
        245,
        "Rollout Output",
        ("H5: actions, qpos/qvel, rewards, success", "MP4: head/wrist/follower videos", "used for analysis or training"),
        ORANGE,
        ORANGE_DARK,
    )

    for box in input_boxes + [optional, benchmark, runner, sim, output]:
        draw_box(draw, box)

    bus_y = 460
    input_centers = [box.x + box.w // 2 for box in input_boxes]
    draw.line((input_centers[0], bus_y, input_centers[-1], bus_y), fill=LINE, width=4)
    for box in input_boxes:
        x = box.x + box.w // 2
        draw.line((x, box.y + box.h, x, bus_y), fill=LINE, width=4)
    draw_polyline_arrow(draw, [(benchmark.x + benchmark.w // 2, bus_y), edge_point(benchmark, "top")])

    draw_polyline_arrow(draw, [edge_point(optional, "right"), (665, optional.y + optional.h // 2), (665, benchmark.y + benchmark.h // 2), edge_point(benchmark, "left")])
    draw_arrow(draw, edge_point(benchmark, "right"), edge_point(runner, "left"))
    draw_polyline_arrow(draw, [edge_point(runner, "bottom"), (1740, 850), (780, 850), edge_point(sim, "top")])
    draw_arrow(draw, edge_point(sim, "right"), edge_point(output, "left"))

    save(image, "diagram_01_molmospaces_benchmark_flow.png")


def render_rby1_benchmark_episode_overview() -> None:
    image, draw = new_canvas(2300, 1450)
    draw_title(
        draw,
        "RBY1 Benchmark And Episode Overview",
        "Choose a benchmark folder first, then run individual fixed episodes inside it.",
    )

    draw_table(
        draw,
        80,
        180,
        [420, 500, 455, 535, 270],
        [90, 150, 150, 150, 150],
        ["Benchmark", "Task Type", "Planner Family", "Episode Result Means", "Current Use"],
        [
            [
                "pick_benchmark",
                "Pick up rigid object",
                "Rigid object grasp planner",
                "One object + one instruction + success/fail video/H5",
                "used now",
            ],
            [
                "pnp_benchmark",
                "Pick and place object",
                "Rigid object grasp + place planner",
                "Pickup object, place target, full trajectory result",
                "available",
            ],
            [
                "opening_benchmark",
                "Open drawer/cabinet/refrigerator",
                "Articulated opening planner",
                "Opening amount and success threshold per episode",
                "available",
            ],
            [
                "door_opening_benchmark",
                "Push or pull door open",
                "Door opening planner",
                "Door joint motion and push/pull success per episode",
                "available",
            ],
        ],
    )

    note = Box(
        155,
        905,
        870,
        290,
        "Episode-Level Output",
        (
            "benchmark idx selects one fixed episode",
            "example columns: idx, house, instruction, success_any, success_last",
            "saved result: H5 trajectory plus MP4 camera videos",
        ),
        BLUE,
        BLUE_DARK,
    )
    distinction = Box(
        1275,
        905,
        870,
        290,
        "Important Distinction",
        (
            "RBY1 is the robot platform, not one task",
            "different benchmark folders use different task/planner logic",
            "pick tasks use grasp candidates; door/opening tasks use articulated logic",
        ),
        PURPLE,
        PURPLE_DARK,
    )
    draw_box(draw, note)
    draw_box(draw, distinction)

    save(image, "diagram_02_rby1_benchmark_episode_overview.png")


def render_molmospace_to_molmobot_flow() -> None:
    image, draw = new_canvas(2700, 1840)
    draw_title(
        draw,
        "MolmoSpaces Planner Data To MolmoBot Learned Policy",
        "MolmoSpaces can generate expert demonstrations; MolmoBot learns from the resulting H5 + MP4 trajectories.",
    )

    lane_y = 165
    lane_h = 1290
    lanes = [
        (70, 760, "MolmoSpaces Planner / Data Generation", GREEN_DARK),
        (970, 760, "Training Dataset", ORANGE_DARK),
        (1870, 760, "MolmoBot Learned Policy", PURPLE_DARK),
    ]
    for x, w, label, color in lanes:
        draw.rounded_rectangle((x, lane_y, x + w, lane_y + lane_h), radius=30, fill=(255, 255, 255), outline=(220, 226, 235), width=3)
        draw_centered_text(draw, (x, lane_y + 22, x + w, lane_y + 80), label, FONT_SECTION, color)

    task = Box(
        130,
        300,
        620,
        230,
        "Benchmark / Task Data",
        ("scene, robot, cameras", "target object or articulated target", "language instruction"),
        BLUE,
        BLUE_DARK,
    )
    planner = Box(
        130,
        620,
        620,
        285,
        "MolmoSpaces Planner",
        ("CuRobo or task-specific planner", "rigid pick may use grasp candidates", "door/opening uses articulated target logic"),
        GREEN,
        GREEN_DARK,
    )
    rollout = Box(
        130,
        1030,
        620,
        260,
        "Successful Expert Rollout",
        ("planner actions executed in MuJoCo", "successful trajectories are saved", "failed rollouts may be filtered"),
        GREEN,
        GREEN_DARK,
    )

    demo_data = Box(
        1040,
        405,
        600,
        335,
        "H5 + MP4 Demonstration Data",
        (
            "H5: actions, qpos/qvel, rewards, success flags",
            "MP4: head and wrist camera videos",
            "language: task_description text",
        ),
        ORANGE,
        ORANGE_DARK,
    )
    train = Box(
        1040,
        930,
        600,
        250,
        "Training Samples",
        ("image frames + language", "robot state + future actions", "used by SynthmanipDataset"),
        GRAY,
        GRAY_DARK,
    )

    model = Box(
        1940,
        360,
        620,
        270,
        "MolmoBot Training",
        ("learns from videos/states/actions", "does not receive grasp candidates", "produces checkpoint"),
        PURPLE,
        PURPLE_DARK,
    )
    eval_inputs = Box(
        1940,
        735,
        620,
        270,
        "Runtime Inputs",
        ("camera frames", "language goal", "robot proprioception"),
        BLUE,
        BLUE_DARK,
    )
    action = Box(
        1940,
        1110,
        620,
        250,
        "Predicted Action Chunk",
        ("base, arms, grippers", "executed in MolmoSpaces/MuJoCo", "saved as eval rollout"),
        PURPLE,
        PURPLE_DARK,
    )

    for box in [task, planner, rollout, demo_data, train, model, eval_inputs, action]:
        draw_box(draw, box)

    draw_arrow(draw, edge_point(task, "bottom"), edge_point(planner, "top"))
    draw_arrow(draw, edge_point(planner, "bottom"), edge_point(rollout, "top"))
    draw_polyline_arrow(
        draw,
        [
            edge_point(rollout, "right"),
            (900, rollout.y + rollout.h // 2),
            (900, demo_data.y + demo_data.h // 2),
            edge_point(demo_data, "left"),
        ],
    )
    draw_arrow(draw, edge_point(demo_data, "bottom"), edge_point(train, "top"))
    draw_polyline_arrow(
        draw,
        [
            edge_point(train, "right"),
            (1810, train.y + train.h // 2),
            (1810, model.y + model.h // 2),
            edge_point(model, "left"),
        ],
    )
    draw_arrow(draw, edge_point(model, "bottom"), edge_point(eval_inputs, "top"))
    draw_arrow(draw, edge_point(eval_inputs, "bottom"), edge_point(action, "top"))

    correction = Box(
        420,
        1590,
        1860,
        180,
        "Key Correction",
        (
            "Grasp candidates help generate planner demonstrations upstream; MolmoBot runtime uses learned visual-language-action behavior, not grasp candidate files.",
        ),
        GRAY,
        GRAY_DARK,
    )
    draw_box(draw, correction)

    save(image, "diagram_03_molmospace_to_molmobot_flow.png")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    for old_png in OUTPUT_DIR.glob("*.png"):
        old_png.unlink()

    render_molmospaces_benchmark_flow()
    render_rby1_benchmark_episode_overview()
    render_molmospace_to_molmobot_flow()


if __name__ == "__main__":
    main()
