import argparse
import os
import sys
from typing import Optional, Tuple
from PIL import Image
from transformers import pipeline
import tensorflow as tf

class NSFWScanner:
    def __init__(self):
        self.files_count = 0
        self.nsfw_count = 0
        self.normal_count = 0
        self.suspect_count = 0

    def parse_arguments(self) -> Tuple[str, bool, Optional[str]]:
        parser = argparse.ArgumentParser(description='Scans a directory for NSFW files')
        parser.add_argument('directory', type=str, help='Directory to start scanning from')
        parser.add_argument('-r', '--recursive', action='store_true', help='Recursively scan subdirectories')
        parser.add_argument('-t', '--target', type=str, default=None, help='Target directory for NSFW files')

        args = parser.parse_args()
        return args.directory, args.recursive, args.target

    def scan_directory(self, classifier, directory: str, recursive: bool, target_dir: Optional[str]) -> None:
        if target_dir and not os.path.exists(target_dir):
            return

        print(f"Scanning directory: {directory}")
        items = os.listdir(directory)

        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path) and item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
                self.process_file(classifier, item_path, target_dir)
            elif recursive and os.path.isdir(item_path):
                self.scan_directory(classifier, item_path, recursive, target_dir)

    def process_file(self, classifier, image_path: str, target_dir: Optional[str]) -> None:
        try:
            image = Image.open(image_path)
        except (OSError, Exception) as e:
            print(f'Error opening image: {image_path} - {e}', file=sys.stderr)
            return

        print(f"Checking {image_path}")
        results = classifier(image)
        nsfw_score, normal_score = 0, 0
        for result in results:
            if result['label'] == 'nsfw':
                nsfw_score = result['score']
            if result['label'] == 'normal':
                normal_score = result['score']

        target_path = os.path.join(target_dir, os.path.basename(image_path)) if target_dir else None
        self.analyze_and_move(image_path, nsfw_score, normal_score, target_path)

    def analyze_and_move(self, image_path: str, nsfw_score: float, normal_score: float, target_path: Optional[str]) -> None:
        self.files_count += 1

        if nsfw_score > 0.5:
            self.nsfw_count += 1
            self.move_file(image_path, target_path, f"NSFW content detected. Moving to {target_path}.")
        elif nsfw_score >= 0.15:
            self.nsfw_count += 1
            self.move_file(image_path, target_path, f"Likely NSFW content. Moving to {target_path}.")
        elif nsfw_score > 0.05:
            self.suspect_count += 1
            print(f"{image_path} may contain NSFW content. Manual review suggested.")
        elif normal_score > 0.9:
            self.normal_count += 1
            print(f"{image_path} is classified as safe.")
        else:
            self.suspect_count += 1
            print(f"{image_path} is ambiguous. Manual review suggested.")

    def move_file(self, source: str, destination: Optional[str], message: str) -> None:
        if destination:
            os.rename(source, destination)
            print(message)
        else:
            print(f"{source} {message}")

if __name__ == "__main__":
    scanner = NSFWScanner()
    directory, recursive, target = scanner.parse_arguments()

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
    scanner.scan_directory(classifier, directory, recursive, target)

    # Show statistics
    print("Scan results:")
    print("============================")
    print(f"Number of files scanned: {scanner.files_count}")
    print(f"Number of NSFW files found: {scanner.nsfw_count}")
    print(f"Number of suspect files found: {scanner.suspect_count}")
    print(f"Number of normal files found: {scanner.normal_count}")
