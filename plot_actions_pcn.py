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