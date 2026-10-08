"""Create matched ground-truth/model MP4s for all completed validation episodes."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import tempfile

import h5py
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont


def collect(root):
    rows = []
    for shard in sorted(root.glob('shard_*')):
        for result in json.loads((shard / 'results.json').read_text()):
            if result['success'] is None or result['process_exit']:
                raise ValueError(f'Incomplete result: {result}')
            source = result['source']
            h5 = Path(source['h5_file'])
            with h5py.File(h5) as data:
                name = data[source['traj_key'] + '/obs/sensor_data/head_camera'][()].tobytes().decode().rstrip('\0')
            ground_truth = h5.parent / name
            models = list((shard / f"episode_{result['index']:03d}").rglob('*head_camera*.mp4'))
            if not ground_truth.is_file() or len(models) != 1:
                raise ValueError(f'Missing or ambiguous paired video: {result["index"]}')
            rows.append(dict(result, ground_truth_video=str(ground_truth), model_video=str(models[0])))
    assert len(rows) == 90 and len({r['index'] for r in rows}) == 90
    return sorted(rows, key=lambda r: r['index'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = collect(args.results)
    for label in ('success', 'failure'):
        (args.output / label).mkdir(parents=True, exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    def render(row):
        label = 'success' if row['success'] else 'failure'
        stem = f"episode_{row['index']:03d}_house_{row['house']}_{label}_comparison"
        filename = stem + '.mp4'
        destination = args.output / label / filename
        header = Image.new('RGB', (768, 52), '#17202b')
        draw = ImageDraw.Draw(header)
        draw.text((8, 5), 'Ground truth: expert demonstration', font=font, fill='white')
        draw.text((392, 5), 'Model inference: ' + label.upper(), font=font, fill='#76eb99' if row['success'] else '#ff9999')
        subtitle = f"Episode {row['index']:03d} | house {row['house']} | 1x playback"
        draw.text((8, 28), subtitle, font=font, fill='white')
        draw.text((392, 28), subtitle, font=font, fill='white')
        with tempfile.TemporaryDirectory(prefix='nav_comparison_') as tmp:
            path = Path(tmp) / 'header.png'
            header.save(path)
            filters = '[0:v]fps=10,scale=384:216:force_original_aspect_ratio=decrease,pad=384:216:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS[left];[1:v]fps=10,scale=384:216:force_original_aspect_ratio=decrease,pad=384:216:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS[right];[left][right]hstack=shortest=0[clip];[clip]pad=768:268:0:52[base];[base][2:v]overlay=0:0:repeatlast=1[labeled];[labeled]format=yuv420p[video]'
            encoding = ['-map', '[video]', '-r', '10', '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20', '-threads', '2', '-movflags', '+faststart']
            subprocess.run([ffmpeg, '-v', 'error', '-y', '-threads', '2', '-filter_complex_threads', '1',
                            '-i', row['ground_truth_video'], '-i', row['model_video'], '-i', str(path),
                            '-filter_complex', filters, *encoding, str(destination)], check=True)
        row['mp4'] = f'{label}/{filename}'
        print(f"Created {row['mp4']}", flush=True)
        return row
    print('Verified matched source videos for all 90 episodes; rendering...', flush=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(render, rows))
    import decord
    for row in rows:
        video = decord.VideoReader(str(args.output / row['mp4']), num_threads=1)
        assert len(video) > 1 and abs(video.get_avg_fps() - 10) < 0.1
        video.get_batch([0, len(video) - 1])
        del video
    (args.output / 'manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
    text = '# Full validation: ground truth and model comparisons\n\n90 held-out trajectory starts from evaluation 6687763: 28 successes, 62 failures (31.1%). Left: the matching expert demonstration. Right: learned-model inference. Each MP4 shows the complete paired head-camera videos at 1x speed and 10 fps. The shorter video holds its last frame until the longer video ends; elapsed durations can differ. Labels report simulator door-opening task success, not grasp-quality classification or process status.\n\n'
    for label in ('success', 'failure'):
        selected = [r for r in rows if ('success' if r['success'] else 'failure') == label]
        lines = [f"[Episode {r['index']:03d}, house {r['house']} (MP4)]({Path(r['mp4']).name})" for r in selected]
        (args.output / label / 'README.md').write_text(f'# {label.title()} comparisons ({len(selected)})\n\n' + '\n\n'.join(lines) + '\n')
        text += f'[{label.title()} comparisons ({len(selected)})]({label}/README.md)\n\n'
    (args.output / 'README.md').write_text(text)
    print('Verified all 90 MP4 comparisons.', flush=True)


if __name__ == '__main__':
    main()
