from geographiclib.geodesic import Geodesic
from statistics import mean

import numpy as np
import argparse
import gpxpy
import tqdm
import cv2
import csv
import sys
import os

def get_corresponding_location(frame_num, frame_count, coordinates):
  """ Returns the coordinate data corresponding to a frame number """
  coor_to_frames = float(len(coordinates)) / frame_count  # ratio of gps length to frame count
  coor_index = int(frame_num * coor_to_frames)            # converts frame number to coordinate index
  return coordinates[coor_index]

def distance(gps1, gps2):
  geod = Geodesic.WGS84  # define the WGS84 ellipsoid
  g = geod.Inverse(float(gps1[0]), float(gps1[1]), float(gps2[0]), float(gps2[1]))
  distance = g['s12']
  return distance 


def main():
    parser = argparse.ArgumentParser(description="Action plotter")
    parser.add_argument('--input', '-i', type=str, help='Input directory the pkl,gpx and video directory is held.')


if __name__ == '__main__':
  main()