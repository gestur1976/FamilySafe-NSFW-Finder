import argparse
import os
import shutil
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
    parser.add_argument('-d', '--frame_delay', type=int, default=None, help='This option sets the time gap (in '
                                                                            'seconds) between each frame analyzed '
                                                                            'from video files. The default is 5.0 '
                                                                            'seconds, meaning only one frame frame '
                                                                            'will be processed every five seconds. '
                                                                            'Set this to 0 if you want to analyze all '
                                                                            'frames.')
    parser.add_argument('-f', '--frame_frequency', type=int, default=None, help='This option controls how often '
                                                                                'frames are selected for analysis in '
                                                                                'a video file. A value of 1 means '
                                                                                'analyzing every single frame. If not '
                                                                                'specified, the default behavior is '
                                                                                'to analyze a frame every 5 seconds.')

    args = parser.parse_args()
    return args.directory, args.recursive, args.target, args.frame_delay, args.frame_frequency


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

    def scan_directory(self, image_classifier, source_dir: str, do_recurse: bool, target_dir: Optional[str], frame_delay: float,
                       frame_frequency: int) -> None:
        if target_dir and not os.path.exists(target_dir):
            return

        print(f"Scanning directory: {source_dir}")
        items = os.listdir(source_dir)

        for item in items:
            item_path = os.path.join(source_dir, item)
            if os.path.isfile(item_path):
                if item.lower().endswith(self.image_extensions):
                    self.process_image_file(image_classifier, item_path, target_dir)
                elif item.lower().endswith(self.video_extensions):
                    self.process_video_file(image_classifier, item_path, target_dir, frame_delay, frame_frequency)
            elif do_recurse and os.path.isdir(item_path):
                self.scan_directory(image_classifier, item_path, do_recurse, target_dir, frame_delay, frame_frequency)

    def process_image_file(self, image_classifier, image_path: str, target_dir: Optional[str]) -> None:
        try:
            image = Image.open(image_path)
        except (OSError, Exception) as e:
            print(f'Error opening image: {image_path} - {e}', file=sys.stderr)
            return

        print(f"Checking {image_path}")
        self.files_count += 1
        if image_classifier.analyze_image(image):
            target_path = os.path.join(target_dir, os.path.basename(image_path)) if target_dir else None
            self.move_file(image_path, target_path, f"NSFW content detected. Moving to {target_path}.")
            self.nsfw_count += 1
        else:
            print(f"{image_path} is classified as safe.")
            self.normal_count += 1

    def process_video_file(self, image_classifier, video_path: str, target_dir: Optional[str], frame_delay: float,
                       frame_frequency: int) -> None:
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)  # Get fps of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if fps == 0:
                print(f"Error: Unable to determine frames per second for {video_path}")
                return
            if frame_delay:
                frame_frequency = int(frame_delay * fps)
            is_nsfw = False

            print(f"Checking {video_path}")
            self.files_count += 1
            frames_to_analyze = range(0, frame_count, frame_frequency)
            for frame_no in frames_to_analyze:
                # Show current progress and percent total
                print(f"Analyzing frame {frame_no}/{frame_count} ({frame_no/frame_count*100:.2f}%)", end='\r')
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = cap.read()
                if not ret:
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(frame)
                if image_classifier.analyze_image(image):
                    is_nsfw = True
                    break
            cap.release()

            if is_nsfw:
                target_path = os.path.join(target_dir, os.path.basename(video_path)) if target_dir else None
                self.move_file(video_path, target_path, f"NSFW content detected. Moving to {target_path}.")
                self.nsfw_count += 1
            else:
                print(f"{video_path} is classified as safe.")
                self.normal_count += 1

        except (cv2.error, Exception) as e:
            print(f'Error processing video: {video_path} - {e}', file=sys.stderr)
            return

    @staticmethod
    def move_file(source: str, destination: Optional[str], message: str) -> None:
        if destination:
            if shutil.copy2(source, destination):   # copy2 preserves metadata
                os.remove(source)
                print(f"{source} {message}")
            else:
                print(f"Failed to move {source} to {destination}", file=sys.stderr)


if __name__ == "__main__":
    directory, recursive, target, frame_delay, frame_frequency = parse_arguments()

    if not os.path.exists(directory):
        print(f'Source directory does not exist: {directory}', file=sys.stderr)
        sys.exit(1)

    if target and not os.path.exists(target):
        print(f'Target directory does not exist: {target}', file=sys.stderr)
        sys.exit(1)

    if frame_delay and frame_frequency:
        print("Both delay and frequency cannot be specified at the same time.", file=sys.stderr)
        sys.exit(1)

    if not frame_frequency and not frame_delay:
        frame_delay = 5.0

    classifier = Classifier("Falconsai/nsfw_image_detection")
    finder = NSFWFinder()

    finder.scan_directory(classifier, directory, recursive, target, frame_delay, frame_frequency)

    # Show statistics
    print("Scan results:")
    print("============================")
    print(f"Number of files scanned: {finder.files_count}")
    print(f"Number of NSFW files found: {finder.nsfw_count}")
    print(f"Number of normal files found: {finder.normal_count}")
