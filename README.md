*Note! The repository is a work in progress. Users to exercise their own discretion when attempting to use any part of this repository.*

# Introduction

The repository provides the scripts to:
1. Extract gpx data from GoPro videos.
2. Detect human actions within GoPro videos.
3. Outputs the actions detected into a CSV file that can be plotted within ArcGIS. 

Scripts to extract telemetry from GoPro videos are adapted from [Jin-Zhe](https://github.com/jin-zhe)'s [gopro](https://github.com/jin-zhe/gopro) repository. Scripts used for action detection are adapted from [oulutan](https://github.com/oulutan)'s [ACAM_Demo](https://github.com/oulutan/ACAM_Demo) repository.

## Environment Setup

1. This project uses [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for package management. The required python packages for GPU implementation can be found in [acam-3.6-gpu.yml](/_env_setup/acam-3.6-gpu.yml).

2. CUDA Version 10.0.130 and CUDNN 7.6.3 has been verified to work with the repository. *Note! Do not install the nvidia driver that comes with the runfile installer. Use the latest graphics driver that nvidia provides.    

3. Follow the installation instructions at oulutan's [repository](https://github.com/oulutan/ACAM_Demo) to setup the necessary environment and dependencies. 

4. Install Golang. (documentation WIP)

## Usage

The repository is currently configured to work using the following directory structure.

![Required directory structure](/_images/folder_structure.png)

GoPro videos that are to be processed must be stored in the folder *video_src*. 

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
```
For every GoPro video in the *video_src* folder, the script extracts the embedded telemetry data and stores it in .gpx format within the *gpx* folder.


To run *plot_actions_pcn.py*, use:

```python
python plot_actions_pcn.py -f <path_to_pkl_folder>
```

For every pkl file within the *pkl* folder, the script will output a csv file in the *csv* and *csv_conv* folders. The csv files contains the longitude and latitude of the detections and can be directly plotted within ArcGIS using the geoprocessing tool *XY to Point*.

The *csv* folder stores the detections at every unique point that is logged within the telemetry data, while *csv_conv* stores the average detections at 10 meter intervals.

