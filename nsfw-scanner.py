import argparse
from typing import Optional
from PIL import Image
from transformers import pipeline
import sys
import os
import shutil


def parse_args() -> tuple[str, bool, Optional[str]]:
    parser = argparse.ArgumentParser(
        description='Scans a file tree for NSFW files')

    parser.add_argument('directory', type=str,
                        help='Directory to start scanning from')
    parser.add_argument('-r', '--recursive', action='store_true',
                        help='Recursively scan subdirectories')
    parser.add_argument('-t', '--target', type=str, default=None,
                        help='Target directory where NSFW files will be moved')

    args = parser.parse_args()
    return args.directory, args.recursive, args.target

# Function to scan a directory for images


def scan_directory(classifier: pipeline, directory: str, recursive: bool, target: Optional[str]) -> None:
    # Check if target directory exists
    if target:
        if not os.path.exists(target):
            return
    print("Scanning directori: " + directory)
    # Get all files in directory
    files = os.listdir(directory)

    for file in files:
        # Check if file is an image
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.gif')):
            # Get image path
            image_path = os.path.join(directory, file)
            # Load image
            try:
                image = Image.open(image_path)
            except OSError:
                print(f'Error opening image: {image_path}', sys.stderr)
                continue
            # Check if image is NSFW
            print("Checking " + image_path)
            result = classifier(image)
            for result_dict in result:
                if result_dict['label'] == 'nsfw':
                    nsfw = result_dict['score']
                if result_dict['label'] == 'normal':
                    normal = result_dict['score']

            # Move image to target directory if NSFW
            if nsfw > 0.05 and nsfw < 0.15:
                print(image_path + ' MAY have NSFW content with a mixed score of nsfw: ' +
                      str(nsfw) + ' and normal: ' + str(normal) + '. A manual review is suggested.')
            elif nsfw >= 0.15 and nsfw < 0.5:
                if target:
                    target_path = os.path.join(target, file)
                    # We move the file to the target path
                    os.rename(image_path, target_path)

                    print(f"{image_path} very likely has NSFW content " + str(nsfw) +
                          " vs " + str(normal) + ". Moved to " + target_path + ".")
                else:
                    print(f"{image_path} very likely has NSFW content " +
                          str(nsfw) + " vs " + str(normal) + ".")
            elif nsfw >= 0.5:
                if target:
                    target_path = os.path.join(target, file)
                    os.rename(image_path, target_path)
                    print(f"{image_path} has NSFW content. Moving to {
                          target_path}.")
                else:
                    print(f"{image_path} has NSFW content.")
            elif normal > 0.9:
                print(image_path + ' is ok.')
            else:
                print(image_path + " is ambiguous " + str(nsfw) +
                      " vs " + str(normal) + ". A manual review is suggested.")

    # If recursive is set to True, scan subdirectories
    if recursive:
        for subdir in os.listdir(directory):
            subdir_path = os.path.join(directory, subdir)
            if os.path.isdir(subdir_path):
                scan_directory(classifier, subdir_path, recursive, target)


# Parse command line arguments
directory, recursive, target = parse_args()

# Check if source directory exists
if not os.path.exists(directory):
    print(f'Source directory does not exist: {directory}', sys.stderr)
    exit(1)

# If specified, check if target directory exists
if target:
    if not os.path.exists(target):
        print(f'Target directory does not exist: {target}', sys.stderr)
        exit(1)

# Load the model
classifier = pipeline('image-classification',
                      model='Falconsai/nsfw_image_detection')

scan_directory(classifier, directory, recursive, target)
