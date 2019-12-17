from geographiclib.geodesic import Geodesic
from statistics import mean

import pickle as pkl
import numpy as np
import argparse
import gpxpy
import tqdm
import cv2
import csv
import sys
import os

GEOD = Geodesic.WGS84 # define the WGS84 ellipsoid
PROXIMITY_THRESHOLD = 10
COLUMN_TITLES = ['Latitude', 'Longitude', 'run/jog', 'sit', 'stand', 'walk', 'ride']

def main(pkl_folder_path=""):
  parser = argparse.ArgumentParser(description="Takes in the detection files in .pkl, coordinates in .gpx and outputs a .csv file.")
  parser.add_argument('-s', '--pkl_path', type=str, required=False, default="")
  parser.add_argument('-f', '--folder_path', type=str, required=False, default="")
  args = parser.parse_args()
  
  if pkl_folder_path != "":
    args.folder_path = pkl_folder_path

  if args.pkl_path:
    pkl_path = args.pkl_path
    plot_actions(pkl_path)
  if args.folder_path:
    folder_path = args.folder_path 
    batch_plot_actions(folder_path)

def process_gpx(gpx_path):
  """ Read in GPX file from given path and process its data """
  with open(gpx_path, 'r') as f:
    gpx = gpxpy.parse(f)

  # GoPro GPX has only one track segment within a single track
  trk = gpx.tracks[0]
  trkseg = trk.segments[0]

  # 1. Remove trkpts without timestamp
  track_points = list(filter(lambda trkpt: trkpt.time is not None, trkseg.points))
  # 2. Sort trkpts chronologically
  track_points.sort(key=lambda trkpt: trkpt.time)
  # 3. Mark initial noisy coordinates attributed by GPS location fixing
  #--------------------------------------------------------------------------------------------to implement function to verify invalid GPS coordinate
  # 4. Format into coordinates
  coordinates = [[x.latitude, x.longitude, x.time] if x else x for x in track_points]

  return coordinates

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

def load_pickle(pickle_path):
  """Loads pickle from given pickle path"""
  return pkl.load(open(pickle_path, 'rb'))

def write_csv(rows, output_path, titles=None, delimiter=','):
  """Writes out rows to csv file given output path"""
  mode = 'w' 
  with open(output_path, mode) as csvfile:
    out_writer = csv.writer(csvfile, delimiter=delimiter)
    if titles:
      out_writer.writerow(titles)
    for row in rows:
      out_writer.writerow(row)

def plot_csv(rows, output_path):
  write_csv(rows, output_path, titles=COLUMN_TITLES)

def convolve_to_tail(plot_points):
  '''plot_points is expected to be in the format [lat,long,run/jog,sit,stand,walk,ride]'''

  head_gps = None
  plot_points_out = []
  scores = []

  for point in plot_points:
    if head_gps is None: #first case
      head_gps = [point[0],point[1]]
      scores.append([point[2],point[3],point[4],point[5],point[6]])

    else: #middle cases 
      tail_gps = [point[0],point[1]]
      scores.append([point[2],point[3],point[4],point[5],point[6]])

      if distance(head_gps,tail_gps) > PROXIMITY_THRESHOLD:
        averages = [mean(x) for x in zip(*scores)]
        plot_points_out.append(tail_gps + averages)
        head_gps = None # reset
        scores = [] # reset

    #last case
  if head_gps:
    averages = [mean(x) for x in zip(*scores)]
    plot_points_out.append(tail_gps + averages)

  return plot_points_out

def convolve_to_head(plot_points):
  '''plot_points is expected to be in the format [lat,long,run/jog,sit,stand,walk,ride]'''
 
  head_gps = None
  plot_points_out = []
  scores = []

  for point in plot_points:
    if head_gps is None: #first case
        head_gps = [point[0],point[1]]
        scores.append([point[2],point[3],point[4],point[5],point[6]])

    else: #middle cases 
      tail_gps = [point[0],point[1]]
      scores.append([point[2],point[3],point[4],point[5],point[6]])

      if distance(head_gps,tail_gps) > PROXIMITY_THRESHOLD:
        averages = [mean(x) for x in zip(*scores)]
        plot_points_out.append(head_gps + averages)
        head_gps = None # reset
        scores = [] # reset

      #last case
  if head_gps:
    averages = [mean(x) for x in zip(*scores)]
    plot_points_out.append(head_gps + averages)
  
  return plot_points_out

def batch_plot_actions(folder_path):
  for f in os.listdir(folder_path):
    pkl_path = os.path.join(folder_path,f)
    plot_actions(pkl_path)
  print("Complete.") 

def plot_actions(pkl_path):
  '''Outputs a csv file of the detections and its corresponding gps coordinate'''
  
  basename = os.path.basename(pkl_path).split('.')[0]
  root_dir = os.path.split(os.path.dirname(pkl_path))[0]

  in_gpx_path = "{}/gpx/{}.MP4.gpx".format(root_dir,basename)
  out_csv_path = "{}/csv/{}.MP4.csv".format(root_dir,basename)
  out_csv_conv_path = "{}/csv_conv/{}.MP4.conv.csv".format(root_dir,basename)
 
  video_detections = load_pickle(pkl_path)
  frame_count = len(video_detections)

  coordinates = process_gpx(in_gpx_path)
  '''[lat,long,timestamp]'''

  plot_points= []

  for frame in range(0,frame_count-1):
    if video_detections[frame]: # appends location every X frames # X is set as ACTION_FREQ in detect_actions_pcn.py
      detection_location = get_corresponding_location(frame, frame_count, coordinates)
      vd = video_detections[frame]
      plot_points.append([detection_location[0],detection_location[1],vd['run/jog'],vd['sit'],vd['stand'],vd['walk'],vd['ride']])
    #[1.3025233, 103.9192801, {'run/jog': 0, 'sit': 0, 'stand': 0, 'walk': 0, 'ride': 0}]

  #unconvolved plot_points
  plot_csv(plot_points, out_csv_path)
    
  #convolve the detections into points within the PROXIMITY_THRESHOLD
  convolved_plot_points = convolve_to_head(plot_points)
  plot_csv(convolved_plot_points,out_csv_conv_path)


if __name__ == '__main__':
  main()