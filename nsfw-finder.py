import argparse
import os
import sys
from typing import Optional, Tuple
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
import tensorflow as tf
import cv2


def parse_arguments() -> Tuple[str, bool, Optional[str]]:
    parser = argparse.ArgumentParser(description='Scans a directory for NSFW files')
    parser.add_argument('directory', type=str, help='Directory to start scanning from')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively scan subdirectories')
    parser.add_argument('-t', '--target', type=str, default=None, help='Target directory for NSFW files')

    args = parser.parse_args()
    return args.directory, args.recursive, args.target


class NSFWFinder:
    def __init__(self):
        self.files_count = 0
        self.nsfw_count = 0
        self.normal_count = 0
        self.suspect_count = 0
        self.image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp')
        self.video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.mpeg', '.mpg', '.m4v')

    def scan_directory(self, classifier, directory: str, recursive: bool, target_dir: Optional[str]) -> None:
        if target_dir and not os.path.exists(target_dir):
            return

        print(f"Scanning directory: {directory}")
        items = os.listdir(directory)

        for item in items:
            item_path = os.path.join(directory, item)
            if os.path.isfile(item_path):
                if item.lower().endswith(self.image_extensions):
                    self.process_image_file(classifier, item_path, target_dir)
                elif item.lower().endswith(self.video_extensions):
                    self.process_video_file(classifier, item_path, target_dir)
            elif recursive and os.path.isdir(item_path):
                self.scan_directory(classifier, item_path, recursive, target_dir)

    def process_image_file(self, classifier, image_path: str, target_dir: Optional[str]) -> None:
        try:
            image = Image.open(image_path)
        except (OSError, Exception) as e:
            print(f'Error opening image: {image_path} - {e}', file=sys.stderr)
            return

        print(f"Checking {image_path}")

        inputs = classifier(image=image, return_tensors="pt").to(device)
        with torch.no_grad():
            results = model(**inputs)

        nsfw_score, normal_score = 0, 0
        for result in results:
            if result['label'] == 'nsfw':
                nsfw_score = result['score']
            if result['label'] == 'normal':
                normal_score = result['score']

        target_path = os.path.join(target_dir, os.path.basename(image_path)) if target_dir else None
        self.analyze_and_move(image_path, nsfw_score, normal_score, target_path)

    def process_video_file(self, classifier, video_path: str, target_dir: Optional[str]) -> None:
        try:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_interval = max(1, frame_count // 10)  # Analyze 10 frames equally spaced
            frames_to_analyze = [i * frame_interval for i in range(10)]

            nsfw_scores = []
            normal_scores = []

            for frame_no in frames_to_analyze:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                results = classifier(image)
                for result in results:
                    if result['label'] == 'nsfw':
                        nsfw_scores.append(result['score'])
                    if result['label'] == 'normal':
                        normal_scores.append(result['score'])

            cap.release()

            avg_nsfw_score = sum(nsfw_scores) / len(nsfw_scores) if nsfw_scores else 0
            avg_normal_score = sum(normal_scores) / len(normal_scores) if normal_scores else 0

            target_path = os.path.join(target_dir, os.path.basename(video_path)) if target_dir else None
            self.analyze_and_move(video_path, avg_nsfw_score, avg_normal_score, target_path)
        except (cv2.error, Exception) as e:
            print(f'Error processing video: {video_path} - {e}', file=sys.stderr)
            return

    def analyze_and_move(self, file_path: str, nsfw_score: float, normal_score: float,
                         target_path: Optional[str]) -> None:
        self.files_count += 1

        if nsfw_score > 0.5:
            self.nsfw_count += 1
            self.move_file(file_path, target_path, f"NSFW content detected. Moving to {target_path}.")
        elif nsfw_score >= 0.15:
            self.nsfw_count += 1
            self.move_file(file_path, target_path, f"Likely NSFW content. Moving to {target_path}.")
        elif nsfw_score > 0.05:
            self.suspect_count += 1
            print(f"{file_path} may contain NSFW content. Manual review suggested.")
        elif normal_score > 0.9:
            self.normal_count += 1
            print(f"{file_path} is classified as safe.")
        else:
            self.suspect_count += 1
            print(f"{file_path} is ambiguous. Manual review suggested.")

    @staticmethod
    def move_file(source: str, destination: Optional[str], message: str) -> None:
        if destination:
            os.rename(source, destination)
            print(message)
        else:
            print(f"{source} {message}")


if __name__ == "__main__":
    finder = NSFWFinder()
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

    model = AutoModelForImageClassification.from_pretrained("Falconsai/nsfw_image_detection")
    classifier = ViTImageProcessor.from_pretrained('Falconsai/nsfw_image_detection')
    classifier.eval()
    device = torch.device(
        'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
    classifier.to(device)

    finder.scan_directory(classifier, directory, recursive, target)

    # Show statistics
    print("Scan results:")
    print("============================")
    print(f"Number of files scanned: {finder.files_count}")
    print(f"Number of NSFW files found: {finder.nsfw_count}")
    print(f"Number of suspect files found: {finder.suspect_count}")
    print(f"Number of normal files found: {finder.normal_count}")
