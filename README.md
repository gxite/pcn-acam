# Introduction

This repository contains scripts that are used on a research project titled *Measuring Physical Profile and Use of Park Connector Network with Deep Learning
and Multi-Source Multi-Modal Data Analytics*, conducted at the National University of Singapore (NUS), Department of Architecture.

The repository provides the scripts to:
1. Extract gpx data from GoPro videos.
2. Detect human actions within GoPro videos.
3. Outputs the actions detected into a CSV file that can be plotted within ArcGIS. 

This repository is adapted and modified from scripts authored by [Jin-Zhe](https://github.com/jin-zhe). The scripts used for action detection is adapted from [oulutan](https://github.com/oulutan)'s repository [ACAM_Demo](https://github.com/oulutan/ACAM_Demo).

## Environment Setup

1. This project uses [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for package management. The required python packages for GPU implementation can be found in [acam-3.6-gpu.yml](/_env_setup/acam-3.6-gpu.yml).

2. CUDA Version 10.0.130 and CUDNN 7.6.3 has been tested to work. *Note! Do not install the nvidia driver that comes with the runfile installer. Use the latest graphics driver that nvidia provides.    

3. Follow the installation instructions at oulutan's [repository](https://github.com/oulutan/ACAM_Demo) to setup the necessary environment and dependencies. 

4. Install Golang. (documentation WIP)

## Usage

The repository is currently configured to work using the following directory structure.

![Required directory structure](/_images/folder_structure.png)

GoPro videos that are to be processed must be stored in the folder *video_src*. 

To analyse a single video:

```python
pyhon detect_actions_pcn.py -v <video path>
```





