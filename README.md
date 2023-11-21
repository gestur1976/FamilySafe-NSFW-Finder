# NSFW Image Scanner

## Description
This script scans a specified directory for images and uses a classifier model to detect NSFW (Not Safe For Work) content. If NSFW content is detected, the script can optionally move these images to a target directory. It supports scanning directories recursively.

## Features
- Scans specified directories for image files (JPG, JPEG, PNG, GIF).
- Utilizes a machine learning model to classify images as NSFW or safe.
- Moves detected NSFW images to a specified target directory.
- Option to scan directories recursively.

## Requirements
- Python 3.x
- PIL (Python Imaging Library)
- Transformers Library
- A pre-trained NSFW classification model (default: Falconsai/nsfw_image_detection)

## Installation


1. Ensure Python 3.x is installed on your system.
2. Clone this repository or download the script.
3. Create a Conda environment and activate it:

```bash
conda create -n nsfw_scanner python=3.9
conda activate nsfw_scanner
```
4. Set up conda environment:

```bash
conda install cuda-toolkit cudnn tensorflow -c "nvidia/label/cuda-11.8.0"
conda env config vars set LD_LIBRARY_PATH=$CONDA_PREFIX/lib
conda deactivate
conda activate nsfw_scanner
```
4. Install the required Python packages:
```bash
pip install -r requirements.txt
```

## Usage
Before using the script activate the conda environment if it isn't activated.

Run the script from the command line, specifying the source directory and optionally the destination folder to move the detected images:

```bash
conda activate nsfw_scanner

python nsfw-scanner.py [directory] [options]
``````

## Arguments
- directory: The directory to start scanning from.
- -r, --recursive: Recursively scan subdirectories.
- [ -t, --target directory ]: Target directory where NSFW files will be moved (optional).

## Example
```bash
python nsfw-scanner.py ~/Images --recursive --target ~/.nsfw_images
```

## Note

- The accuracy of NSFW detection depends on the model used.
- The script requires internet access to download the model upon first run.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contribution

Contributions to the project are welcome. Please fork the repository and submit a pull request with your changes.

## Disclaimer

This tool is for educational purposes only. The author is not responsible for misuse or for any damage that may occur.
