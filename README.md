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
conda create -n nsfw_finder python=3.9
conda activate nsfw_finder
```
4. Set up the conda environment:

```bash
conda install cuda-toolkit cudnn tensorflow -c "nvidia/label/cuda-11.8.0"
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda deactivate
conda activate nsfw_finder
```

5. Install required Python packages:
```bash
pip install -r requirements.txt
```

## Usage

Activate the conda environment before using the script.

Run the script via command line, specifying the source directory and, if desired, a destination folder for detected media files:

```bash
conda activate nsfw_finder
python nsfw-finder.py [directory] [options]
```

## Arguments

- directory: The directory to scan.
- -r, --recursive: Scan subdirectories recursively.
- [-t, --target directory]: Optional target directory for NSFW files.

## Example
```bash
python nsfw-finder.py ~/Media --recursive --target ~/.nsfw_media
or
# to disable NUMA warnings if you get them prepend the following env VAR to your command
TF_CPP_MIN_LOG_LEVEL=3 python nsfw-finder.py ~/Images --recursive --target ~/.nsfw_images
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