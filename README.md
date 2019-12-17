# Introduction

This repository contains scripts that are used on a research project titled *Measuring Physical Profile and Use of Park Connector Network with Deep Learning
and Multi-Source Multi-Modal Data Analytics*, conducted at the National University of Singapore (NUS), Department of Architecture.

The repository provides the scripts to:
1. Extract gpx data from GoPro videos.
2. Detect human actions within the GoPro videos.
3. Outputs the actions detected and the correspodning GPS location into a CSV file that can be plotted within ArcGIS. 

*plot_actions_pcn.py* and *extract_telemetries.py* are adapted from scripts originally authored by [Jin-Zhe](https://github.com/jin-zhe). 

*detect_actions_pcn.py* and its correspondingly dependencies to enable action detection are adapted from [oulutan](https://github.com/oulutan)'s repository [ACAM_Demo](https://github.com/oulutan/ACAM_Demo).

The supporting scripts used for extracting telemetries from GoPRo Videos are based on works by [tkraijina](https://github.com/tkrajina/gpxgo), [stilldavid](https://github.com/stilldavid) and [paulmach](https://github.com/paulmach).The methodology is described in [this](https://community.gopro.com/t5/Cameras/Hero5-Session-Telemetry/m-p/40278/highlight/true#/M20188) forum.

## Environment Setup

1. This project uses [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for package management. The required python packages for GPU implementation can be found in [acam-3.6-gpu.yml](/_env_setup/acam-3.6-gpu.yml).

2. CUDA Version 10.0.130 and CUDNN 7.6.3 has been tested to work. *Note! Do not install the nvidia driver that comes with the runfile installer. Use the latest graphics driver that nvidia provides.    

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


To use *detect_actions_pcn.py*:

```python
#To process single video, run the following and replace <video_path> with the actual video path of the desired file in your machine. The script will only work if the video is kept within the video_src folder as described above.

python detect_actions_pcn.py -v <video_path>

#To process multiple videos, run the following and replace <folder_path> with the location of your video_src folder in your machine.

python detect_actions_pcn.py -f <folder_path>
```

To use *extract_telemetries.py*:

```python
#To process single video, run the following and replace <video_path> with the actual video path of the desired file in your machine. 

python extract_telemetries.py -v <video_path>

#To process multiple videos, run the following and replace <folder_path> with the location of your video_src folder in your machine.

python extract_telemetries.py -f <folder_path>
```




