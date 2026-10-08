"""Apply the existing grasp criterion to model rollouts and retain MP4 comparisons."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile

import decord
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont
from lrl_molmospaces_extensions.dataset.episode import DoorOpeningEpisode
from lrl_molmospaces_extensions.quality_check.opened_without_grasp import OpenedWithoutGraspCheck


def main():
    root = Path(__file__).resolve().parents[1] / 'docs/slide_history/media'
    rows = json.loads((root / 'manifest.json').read_text())
    check = OpenedWithoutGraspCheck(root, min_grasp_frames=30)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    for row in rows:
        path = Path(row['model_video']).parent / 'trajectories_batch_1_of_1.h5'
        episode = DoorOpeningEpisode.from_h5(path)
        assert episode.success == row['success'], row['index']
        row.update(grasp_label=check.label_episode(episode) or 'did_not_open', grasp_frames=episode.grasp_frames,
                   first_grasp_frame=episode.first_grasp_frame, last_grasp_frame=episode.last_grasp_frame,
                   first_success_s=episode.first_success_s, model_h5=str(path), min_grasp_frames=30)
    def relabel(row):
        source = root / row['mp4']
        if row['success']:
            label = row['grasp_label']
            folder = root / 'success' / label
            folder.mkdir(parents=True, exist_ok=True)
            destination = folder / f"episode_{row['index']:03d}_house_{row['house']}_success_{label}_comparison.mp4"
            if source == destination:
                row.pop('gif', None)
                return row
            with tempfile.TemporaryDirectory(prefix='grasp_comparison_') as tmp:
                header = Image.new('RGB', (768, 52), '#17202b')
                draw = ImageDraw.Draw(header)
                draw.text((8, 5), 'Ground truth: expert demonstration', font=font, fill='white')
                title = 'SUCCESS: WITH GRASP' if label == 'opened_with_grasp' else 'SUCCESS: WITHOUT QUALIFYING GRASP'
                draw.text((392, 5), title, font=font, fill='#76eb99' if label == 'opened_with_grasp' else '#ffd27f')
                draw.text((8, 28), f"Episode {row['index']:03d} | house {row['house']} | 1x playback", font=font, fill='white')
                draw.text((392, 28), f"Recorded grasp: {row['grasp_frames']} frames | required: 30", font=font, fill='white')
                png = Path(tmp) / 'header.png'
                header.save(png)
                subprocess.run([ffmpeg, '-v', 'error', '-y', '-threads', '2', '-filter_complex_threads', '1',
                                '-i', str(source), '-i', str(png), '-filter_complex', '[0:v][1:v]overlay=0:0:repeatlast=1[v]',
                                '-map', '[v]', '-r', '10', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                                '-threads', '2', '-movflags', '+faststart', str(destination)], check=True)
            video = decord.VideoReader(str(destination), num_threads=1)
            video.get_batch([0, len(video)-1]); del video
            if source != destination:
                source.unlink()
            row['mp4'] = str(destination.relative_to(root))
        row.pop('gif', None)
        print(f"Episode {row['index']:03d}: {row['grasp_label']} ({row['grasp_frames']} grasp frames)", flush=True)
        return row
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(relabel, rows))
    counts = {label: sum(r['grasp_label'] == label for r in rows) for label in ('opened_with_grasp','opened_without_grasp','did_not_open')}
    assert len(rows) == 90 and sum(counts.values()) == 90
    for gif in root.rglob('*.gif'):
        gif.unlink()
    (root / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    (root / 'grasp_summary.json').write_text(json.dumps({'min_grasp_frames':30,'criterion':'Cumulative frames of either recorded gripper grasp state > 0 or task_info.handle_grasped; same classifier as training dataset. Not a visual certification or a consecutive-duration test.','counts':counts,'opening_success_rate':28/90,'opened_with_grasp_rate':counts['opened_with_grasp']/90}, indent=2)+'\n')
    lines = ['index\thouse\topening_success\tgrasp_label\tgrasp_frames\tfirst_success_s\tvideo']
    for row in rows:
        lines.append(f"{row['index']}\t{row['house']}\t{int(row['success'])}\t{row['grasp_label']}\t{row['grasp_frames']}\t{row['first_success_s']}\t{row['mp4']}")
    (root / 'grasp_index.tsv').write_text('\n'.join(lines)+'\n')
    for label in counts:
        folder = root / 'failure' if label == 'did_not_open' else root / 'success' / label
        selected = [r for r in rows if r['grasp_label'] == label]
        (folder / 'README.md').write_text(f'# {label.replace("_", " ").title()} ({len(selected)})\n\n' + '\n\n'.join(f"[Episode {r['index']:03d}, house {r['house']} — {r['grasp_frames']} grasp frames]({Path(r['mp4']).name})" for r in selected)+'\n')
    (root / 'success/README.md').write_text('# Successful openings: grasp classification\n\n[With qualifying grasp (1)](opened_with_grasp/README.md)\n\n[Without qualifying grasp (27)](opened_without_grasp/README.md)\n')
    (root / 'README.md').write_text('# Full validation comparison videos\n\nAll 90 paired MP4s: expert ground truth on the left, model inference on the right, 10 fps at 1x playback. The shorter clip holds its final frame. GIFs have been removed.\n\n[Opened with qualifying grasp (1)](success/opened_with_grasp/README.md)\n\n[Opened without qualifying grasp (27)](success/opened_without_grasp/README.md)\n\n[Did not open successfully (62)](failure/README.md)\n\nThe existing dataset filter requires at least 30 cumulative recorded grasp frames (about 3 seconds at 10 Hz). Below-threshold does not necessarily mean zero grasp contact. The simulator stops successful rollouts upon task success, which can limit recorded grasp duration. Opening success is 28/90 (31.1%); success with qualifying grasp is 1/90 (1.1%).\n\nMetrics: [grasp index](grasp_index.tsv), [summary](grasp_summary.json), and [source/video manifest](manifest.json).\n')
    print(json.dumps(counts), flush=True)


if __name__ == '__main__':
    main()
