# Introduction

This repository contains scripts that are used on a research project titled *Measuring Physical Profile and Use of Park Connector Network with Deep Learning
and Multi-Source Multi-Modal Data Analytics*, conducted at the National University of Singapore (NUS), Department of Architecture.

The repository provides the scripts to:
1. Extract gpx data from GoPro videos.
2. Detect human actions within the GoPro videos.
3. Outputs the actions detected and the correspodning GPS location into a CSV file that can be plotted within ArcGIS. 

<<<<<<< HEAD
Scripts to extract telemetry from GoPro videos are adapted from [Jin-Zhe](https://github.com/jin-zhe)'s [gopro](https://github.com/jin-zhe/gopro) repository. Scripts used for action detection are adapted from [oulutan](https://github.com/oulutan)'s [ACAM_Demo](https://github.com/oulutan/ACAM_Demo) repository.
=======
*plot_actions_pcn.py* and *extract_telemetries.py* are adapted from scripts originally authored by [Jin-Zhe](https://github.com/jin-zhe). 

*detect_actions_pcn.py* and its correspondingly dependencies to enable action detection are adapted from [oulutan](https://github.com/oulutan)'s repository [ACAM_Demo](https://github.com/oulutan/ACAM_Demo).

The supporting scripts used for extracting telemetries from GoPRo Videos are based on works by [tkraijina](https://github.com/tkrajina/gpxgo), [stilldavid](https://github.com/stilldavid) and [paulmach](https://github.com/paulmach).The methodology is described in [this](https://community.gopro.com/t5/Cameras/Hero5-Session-Telemetry/m-p/40278/highlight/true#/M20188) forum.
>>>>>>> d43173029a2a00363f62e66b32ef5b60635e237b

## Environment Setup

1. This project uses [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for package management. The required python packages for GPU implementation can be found in [acam-3.6-gpu.yml](/_env_setup/acam-3.6-gpu.yml).

2. CUDA Version 10.0.130 and CUDNN 7.6.3 has been verified to work with the repository. *Note! Do not install the nvidia driver that comes with the runfile installer. Use the latest graphics driver that nvidia provides.    

3. Follow the installation instructions at oulutan's [repository](https://github.com/oulutan/ACAM_Demo) to setup the necessary environment and dependencies. 

4. Install Golang. This is required for *extract_telemetries.py* to work. Jin-Zhe has provided a summary in the following [guide](https://github.com/jin-zhe/gopro/tree/a7e563a65dc934515a88a5f2408db674b92a58fc).

## Usage

The repository is currently configured to work using the following directory structure.

![Required directory structure](/_images/folder_structure.png)

The videos that are to be processed MUST be stored in the folder *video_src*, for the existing code to work without modifications.

The following scripts, can be ran individually:

- *detect_actions_pcn.py*
- *extract_telemetries.py* 
- *plot_actions_pcn.py* 

<<<<<<< HEAD
The scripts should be used in the following sequence:
1. *detect_actions_pcn.py*
2. *extract_telemetries.py*
3. *plot_actions_pcn.py*

*plot_actions_pcn.py* requires output files from *detect_actions_pcn.py* and *extract_telemetries.py* to operate.

To run *detect_actions_pcn.py*, use:

```python
python detect_actions_pcn.py -f <path_to_video_src_folder>
```
For every video in the *video_src* folder, the script generates a pkl file (containing the detections), a log file and a modified video file which visualizes the detections. These files are stored in the *pkl*, *log* and *video_vis* folders respectively. 


To run *extract_telemetries.py*, use:

```python
python extract_telemetries.py -f <path_to_video_src_folder>
=======

To use *detect_actions_pcn.py*:

```python
#To process single video, run the following and replace <video_path> with the actual video path of the desired file in your machine. The script will only work if the video is kept within the video_src folder as described above.

python detect_actions_pcn.py -v <video_path>

#To process multiple videos, run the following and replace <folder_path> with the location of your video_src folder in your machine.

python detect_actions_pcn.py -f <folder_path>
>>>>>>> d43173029a2a00363f62e66b32ef5b60635e237b
```
For every GoPro video in the *video_src* folder, the script extracts the embedded telemetry data and stores it in .gpx format within the *gpx* folder.


<<<<<<< HEAD
To run *plot_actions_pcn.py*, use:
=======
To use *extract_telemetries.py*:

```python
#To process single video, run the following and replace <video_path> with the actual video path of the desired file in your machine. 

python extract_telemetries.py -v <video_path>

#To process multiple videos, run the following and replace <folder_path> with the location of your video_src folder in your machine.

python extract_telemetries.py -f <folder_path>
```
>>>>>>> d43173029a2a00363f62e66b32ef5b60635e237b

```python
python plot_actions_pcn.py -f <path_to_pkl_folder>
```

For every pkl file within the *pkl* folder, the script will output a csv file in the *csv* and *csv_conv* folders. The csv files contains the longitude and latitude of the detections and can be directly plotted within ArcGIS using the geoprocessing tool *XY to Point*.

The *csv* folder stores the detections at every unique point that is logged within the telemetry data, while *csv_conv* stores the average detections at 10 meter intervals.

