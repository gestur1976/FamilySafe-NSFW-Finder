import argparse
import os
import sys
from typing import Optional, Tuple
from PIL import Image
import torch
from transformers import AutoModelForImageClassification, ViTImageProcessor
import tensorflow as tf
import cv2


def parse_arguments() -> Tuple[str, bool, Optional[str], Optional[float], Optional[int]]:
    parser = argparse.ArgumentParser(description='Scans a directory for NSFW files')
    parser.add_argument('directory', type=str, help='Directory to start scanning from')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively scan subdirectories')
    parser.add_argument('-t', '--target', type=str, default=None, help='Target directory for NSFW files')
    parser.add_argument('-d', '--delay', type=int, default=5, help='This option sets the time gap (in seconds) '
                                                                   'between each frame analyzed from video files. The '
                                                                   'default is 5.0 seconds, meaning only one out of '
                                                                   'every five seconds will be processed. Set this to '
                                                                   '0 if you want to analyze all frames.')
    parser.add_argument('-f', '--frequency', type=int, help='This option controls how often frames are selected for '
                                                            'analysis in a video file. A value of 1 means analyzing '
                                                            'every single frame. If not specified, the default '
                                                            'behavior is equivalent to using `-d 5`.')

    args = parser.parse_args()
    return args.directory, args.recursive, args.target, args.delay, args.frequency

class Classifier:
    def __init__(self, model_name):
        gpu_devices = tf.config.experimental.list_physical_devices("GPU")
        for device in gpu_devices:
            tf.config.experimental.set_memory_growth(device, True)
        self.model = AutoModelForImageClassification.from_pretrained(model_name)
        self.classifier = ViTImageProcessor.from_pretrained(model_name)
        self.model.eval()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
        self.model.to(self.device)

    def analyze_image(self, image):
        inputs = self.classifier(images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            results = self.model(**inputs)
        log_items = results.logits
        predicted_label = log_items.argmax(-1).item()
        return self.model.config.id2label[predicted_label] == "nsfw"

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
        self.files_count += 1
        if classifier.analyze_image(image_path):
            target_path = os.path.join(target_dir, os.path.basename(image_path)) if target_dir else None
            self.move_file(image_path, target_path, f"NSFW content detected. Moving to {target_path}.")
            self.nsfw_count += 1
        else:
            print(f"{image_path} is classified as safe.")
            self.normal_count += 1

    def process_video_file(self, classifier, video_path: str, target_dir: Optional[str]) -> None:
        try:
            cap = cv2.VideoCapture(video_path)
            frames_per_second = cap.get(cv2.CAP_PROP_FPS)   # Get fps of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if frames_per_second == 0:
                print(f"Error: Unable to determine frames per second for {video_path}")
                return

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

            cap.release()

            avg_nsfw_score = sum(nsfw_scores) / len(nsfw_scores) if nsfw_scores else 0
            avg_normal_score = sum(normal_scores) / len(normal_scores) if normal_scores else 0

            target_path = os.path.join(target_dir, os.path.basename(video_path)) if target_dir else None
            self.analyze_and_move(video_path, avg_nsfw_score, avg_normal_score, target_path)
        except (cv2.error, Exception) as e:
            print(f'Error processing video: {video_path} - {e}', file=sys.stderr)
            return

    @staticmethod
    def move_file(source: str, destination: Optional[str], message: str) -> None:
        if destination:
            os.rename(source, destination)
            print(message)
        else:
            print(f"{source} {message}")


if __name__ == "__main__":
    directory, recursive, target, delay, frequency = parse_arguments()

    if not os.path.exists(directory):
        print(f'Source directory does not exist: {directory}', file=sys.stderr)
        sys.exit(1)

    if target and not os.path.exists(target):
        print(f'Target directory does not exist: {target}', file=sys.stderr)
        sys.exit(1)


    classifier = Classifier("Falconsai/nsfw_image_detection")
    finder = NSFWFinder()

    finder.scan_directory(classifier, directory, recursive, target)

    # Show statistics
    print("Scan results:")
    print("============================")
    print(f"Number of files scanned: {finder.files_count}")
    print(f"Number of NSFW files found: {finder.nsfw_count}")
    print(f"Number of suspect files found: {finder.suspect_count}")
    print(f"Number of normal files found: {finder.normal_count}")
