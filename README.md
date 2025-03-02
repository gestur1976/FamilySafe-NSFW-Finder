# FamilySafe NSFW Finder

## Description
This script scans a directory for images and videos, using a classifier model to detect NSFW (Not Safe For Work) content. If NSFW content is found, the script can optionally move these media files to a specified directory. It supports recursive scanning of directories.

## Features
- Scans specified directories for image and video files (.jpg, .jpeg, .png, .gif, .bmp, .tiff, .tif, .webp, .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm, .mpeg, .mpg, .m4v).
- Uses [Falconsai/nsfw_image_detection](https://huggingface.co/Falconsai/nsfw_image_detection) model to classify images as NSFW or safe.
- Can move detected NSFW media files to a chosen directory.
- Option for recursive directory scanning.

## Requirements
- Python 3.x
- Anaconda or Miniconda (for TensorFlow and Keras support)

## Installation
1. Ensure Python 3.x is installed on your system.
2. Clone the repository or download the script.
3. Create a Conda environment:

```bash
conda create -n nsfw_scanner python=3.10 cuda-toolkit cudnn tensorflow
conda activate nsfw_scanner
```

4. Set up the conda environment:
```bash
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda deactivate
conda activate nsfw_scanner
```

5. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Usage
Activate the conda environment before using the script.

Run the script via command line, specifying the source directory and, if desired, a destination folder for detected media files:

```bash
conda activate nsfw_scanner
python nsfw-scanner.py [directory] [options]
```

## Arguments
- `directory`: The directory to scan.
- `-r`, `--recursive`: Scan subdirectories recursively.
- `-t`, `--target <directory>`: Optional target directory for NSFW files.
- `-d`, `--frame_delay <seconds>`: Set the time gap between each frame analyzed from video files (default is 5 seconds). 
    - A lower value will increase processing speed, but may lead to less accurate results.
    - Use this option if you want to prioritize speed over accuracy or need faster scans for large volumes of videos. 
- `-f`, `--frame_frequency <frames>`: Set the number of frames between each frame analyzed from video files.
    - A higher value will reduce processing time but may decrease accuracy.

## Example
```bash
python nsfw-scanner.py ~/Media --recursive --target ~/.nsfw_media --frame_delay 2.0
```
or with frame delay option:
```bash
# Analyze a video at one frame every 10 seconds (for faster analysis, albeit slightly less accurate)
python nsfw-scanner.py ~/Videos/ -d 10
```
## Notes
- NSFW detection accuracy depends on the model.
- Internet access is required to download the model on first run.

## License
Licensed under the MIT License - see the LICENSE file for details.

## Contributors
* Josep Hervás (https://github.com/gestur1976)
* Ross Fisher (https://github.com/zorrobyte)

Contributions to the project are welcome. Please fork the repository and submit a pull request with your changes.

## Disclaimer
This tool is for educational use only. The authors are not responsible for misuse or any resulting damage.
