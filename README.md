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
conda create -n nsfw_finder python=3.10 cuda-toolkit cudnn tensorflow
conda activate nsfw_finder
```

4. Set up the conda environment:
```bash
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
```
## If you get NUMA errors/warnings with NVIDIA GPUs try the following before running the finder:

### 1. Check Nodes
```bash
lspci | grep -i nvidia
  
01:00.0 VGA compatible controller: NVIDIA Corporation TU106 [GeForce RTX 2060 12GB] (rev a1)
01:00.1 Audio device: NVIDIA Corporation TU106 High Definition Audio Controller (rev a1)
```
The first line shows the address of the VGA compatible device, NVIDIA Geforce, as **01:00** . Each one will be different, so let's change this part carefully.
### 2. Check and change NUMA setting values
If you go to `/sys/bus/pci/devicecs/`, you can see the following list:
```bash
ls /sys/bus/pci/devices/
  
0000:00:00.0  0000:00:06.0  0000:00:15.0  0000:00:1c.0  0000:00:1f.3  0000:00:1f.6  0000:02:00.0
0000:00:01.0  0000:00:14.0  0000:00:16.0  0000:00:1d.0  0000:00:1f.4  0000:01:00.0
0000:00:02.0  0000:00:14.2  0000:00:17.0  0000:00:1f.0  0000:00:1f.5  0000:01:00.1
```
01:00.0 checked above is visible. However, 0000: is attached in front.

### 3. Check if it is connected.
```bash
cat /sys/bus/pci/devices/0000\:01\:00.0/numa_node
  
-1
```
-1 means no connection, 0 means connected.

### 4. Fix it with the command below.
```bash
sudo echo 0 | sudo tee -a /sys/bus/pci/devices/0000\:01\:00.0/numa_node
  
0
```
It shows 0 which means connected!

### 5. Check again:
```bash
cat /sys/bus/pci/devices/0000\:01\:00.0/numa_node
  
0
```
That's it!

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