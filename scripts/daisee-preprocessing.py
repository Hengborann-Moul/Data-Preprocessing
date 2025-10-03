import argparse
import importlib
import os
import sys

from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, os.pardir))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

main = importlib.import_module("main")
process_video_to_npy = getattr(main, "process_video_to_npy")

ROOT_DIR = "../DAiSEE"


def get_video_paths(obj):
    video_paths = []
    obj_dir = os.path.join(ROOT_DIR, f"DataSet/{obj}")
    for root, _, files in os.walk(obj_dir):
        for file in files:
            if file.endswith(".avi") or file.endswith(".mp4"):
                video_paths.append(os.path.join(root, file))
    return video_paths


def get_output_filename(video_path):
    base_name = os.path.basename(video_path)
    name, _ = os.path.splitext(base_name)
    return f"{name}.npy"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process DAiSEE dataset videos into frame tensors"
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=3,
        help="Target frames per second to extract (default: 3)",
    )
    args = parser.parse_args()

    objs = ["Train", "Test", "Validation"]

    # Create Processed directory
    processed_dir = os.path.join(ROOT_DIR, "Processed")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    for obj in tqdm(objs, desc="Processing datasets: "):
        # Create object directory inside Processed
        output_obj_dir = os.path.join(processed_dir, f"{obj}{args.fps}")
        if not os.path.exists(output_obj_dir):
            os.makedirs(output_obj_dir)

        video_paths = get_video_paths(obj)
        print(f"Found {len(video_paths)} videos in {obj} dataset.")
        # print(video_paths[0])
        # print(get_output_filename(video_paths[0]))
        for video_path in tqdm(video_paths, desc=f"Processing {obj} videos: "):
            output_filename = get_output_filename(video_path)
            process_video_to_npy(
                video_path, output_obj_dir, output_filename, target_fps=args.fps
            )
        print(f"Processed {len(video_paths)} videos in {obj} dataset.")
