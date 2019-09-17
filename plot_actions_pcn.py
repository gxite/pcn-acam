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

def main():
    parser = argparse.ArgumentParser(description="Takes in the detection files in .pkl, coordinates in .gpx and outputs a .csv file.")
    parser.add_argument('-v', '--video_path', type=str, required=False, default="")
    parser.add_argument('-f', '--folder_path', type=str, required=False, default="")
    args = parser.parse_args()

    root_dir = args.input
    pkl_dir = os.path.join(root_dir,"pkl")
    gpx_dir = os.path.join(root_dir,"gpx")
    csv_dir = os.path.join(root_dir,"csv") #output dir

    #-------------------------------------------------------sort out flag actions here
    plot_actions(pkl_dir, gpx_dir,csv_dir)


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

def get_filenames_from_dir(directory):
  '''returns basename without any extension'''
  files = [os.path.join(directory,f) for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
  files_out = []

  #removes all the file extensions
  for f in files:
    temp = f
    while(os.path.splitext(temp)[1] != ""):
        temp = os.path.splitext(temp)[0] #removes file extension
    files_out.append(os.path.basename(temp))#appends basename
  return files_out

def set_filename_ext(file_dir,filename,ext):
  '''Valid ext values are: pkl,gpx,xsv'''
  if ext == "pkl":
    with_ext = '{}.ACT.pkl'.format(filename)
    return os.path.join(file_dir,with_ext)
  elif ext == "gpx":
    with_ext = '{}.MP4.gpx'.format(filename)
    return os.path.join(file_dir,with_ext)
  elif ext == "csv":
    with_ext = '{}.ACT.csv'.format(filename)
    return os.path.join(file_dir,with_ext)
  else:
    raise Exception('argument "{}" is not valid'.format(ext))

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

def plot_actions(pkl_dir, gpx_dir, csv_dir):
  '''Outputs a csv file of the detections and its corresponding gps coordinate'''

  #uses the files in the pkl directory to generate a list of filenames that is extension free
  filename_list = get_filenames_from_dir(pkl_dir)

  for filename in filename_list:
    '''Process each pkl and its corresponding gpx.'''
      
    #sets the filepaths for the required files
    pkl_file_path = set_filename_ext(pkl_dir,filename,"pkl")
    gpx_file_path = set_filename_ext(gpx_dir,filename,"gpx")
    csv_file_path = set_filename_ext(csv_dir,filename, "csv")
    csv_convolved_file_path = set_filename_ext(csv_dir,filename+"_convolved", "csv")
    
    video_detections = load_pickle(pkl_file_path)
    frame_count = len(video_detections)

    coordinates = process_gpx(gpx_file_path)
    '''[lat,long,timestamp]'''

    plot_points= []

    for frame in range(0,frame_count-1):
      if video_detections[frame]: # appends location only if there is action detection. ie every 8 frames
        detection_location = get_corresponding_location(frame, frame_count, coordinates)
        vd = video_detections[frame]
        plot_points.append([detection_location[0],detection_location[1],vd['run/jog'],vd['sit'],vd['stand'],vd['walk'],vd['ride']])
      #[1.3025233, 103.9192801, {'run/jog': 0, 'sit': 0, 'stand': 0, 'walk': 0, 'ride': 0}]

    #unconvolved plot_points
    plot_csv(plot_points, csv_file_path)
    
    #convolve the detections into points within the PROXIMITY_THRESHOLD
    convolved_plot_points = convolve_to_head(plot_points)
    plot_csv(convolved_plot_points,csv_convolved_file_path)


if __name__ == '__main__':
  main()