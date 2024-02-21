import argparse
import os
import sys
from typing import Optional, Tuple
from PIL import Image
from transformers import pipeline
import tensorflow as tf

files_count = 0
nsfw_count = 0
normal_count = 0
suspect_count = 0
def parse_arguments() -> Tuple[str, bool, Optional[str]]:
    parser = argparse.ArgumentParser(description='Scans a directory for NSFW files')
    parser.add_argument('directory', type=str, help='Directory to start scanning from')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively scan subdirectories')
    parser.add_argument('-t', '--target', type=str, default=None, help='Target directory for NSFW files')

    args = parser.parse_args()
    return args.directory, args.recursive, args.target

def scan_directory(classifier: pipeline, directory: str, recursive: bool, target_dir: Optional[str]) -> None:
    if target_dir and not os.path.exists(target_dir):
        return

    print(f"Scanning directory: {directory}")
    files = os.listdir(directory)

    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            image_path = os.path.join(directory, file)
            try:
                image = Image.open(image_path)
            except OSError:
                print(f'Error opening image: {image_path}', file=sys.stderr)
                continue

            print(f"Checking {image_path}")
            results = classifier(image)
            nsfw_score, normal_score = 0, 0
            for result in results:
                if result['label'] == 'nsfw':
                    nsfw_score = result['score']
                if result['label'] == 'normal':
                    normal_score = result['score']

            target_path = os.path.join(target_dir, file) if target_dir else None
            analyze_and_move(image_path, nsfw_score, normal_score, target_path)

    if recursive:
        for subdir in [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]:
            scan_directory(classifier, os.path.join(directory, subdir), recursive, target_dir)

def analyze_and_move(image_path: str, nsfw_score: float, normal_score: float, target_path: Optional[str]) -> None:
    global files_count
    global nsfw_count
    global normal_count
    global suspect_count

    files_count += 1

    if nsfw_score > 0.5:
        global nsfw_count
        nsfw_count += 1
        move_file(image_path, target_path, f"NSFW content detected. Moving to {target_path}.")
    elif nsfw_score >= 0.15:
        nsfw_count += 1
        move_file(image_path, target_path, f"Likely NSFW content. Moving to {target_path}.")
    elif nsfw_score > 0.05:
        suspect_count += 1
        print(f"{image_path} may contain NSFW content. Manual review suggested.")
    elif normal_score > 0.9:
        normal_count += 1
        print(f"{image_path} is classified as safe.")
    else:
        suspect_count += 1
        print(f"{image_path} is ambiguous. Manual review suggested.")

def move_file(source: str, destination: Optional[str], message: str) -> None:
    if destination:
        os.rename(source, destination)
        print(message)                                                                      
    else:
        print(f"{source} {message}")

directory, recursive, target = parse_arguments()

if not os.path.exists(directory):
    print(f'Source directory does not exist: {directory}', file=sys.stderr)
    sys.exit(1)

if target and not os.path.exists(target):
    print(f'Target directory does not exist: {target}', file=sys.stderr)
    sys.exit(1)

gpu_devices = tf.config.experimental.list_physical_devices("GPU")
for device in gpu_devices:
    tf.config.experimental.set_memory_growth(device, True)
classifier = pipeline('image-classification', model='Falconsai/nsfw_image_detection')
scan_directory(classifier, directory, recursive, target)
# Show statistics
print("Scan results:")
print("============================")
print(f"Number of files scanned: {files_count}")
print(f"Number of NSFW files found: {nsfw_count}")
print(f"Number of suspect files found: {suspect_count}")
print(f"Number of normal files found: {normal_count}")
