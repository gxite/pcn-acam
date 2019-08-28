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

def mark_null_coordinates(track_points):
  """ Trims off initial <trkpt lat="0" lon="0"> points """
  next_i = 0
  while next_i < len(track_points):
    trkpt = track_points[next_i]
    if trkpt.latitude != 0.0 and trkpt.longitude != 0.0:
      break
    next_i += 1
  
  if next_i:
    # mark coordinates
    for i in range(next_i):
      track_points[i] = False
  
  return next_i

def mark_outliers(start, track_points, qualification_length=770):
  """ Trims off initial outlier points attributed by poor GPS reception """
  next_i = start # start off where mark_null_coordinates() ended
  while next_i < len(track_points) - 1:
    # if current pair is within bounds
    if not is_outlier(track_points[next_i], track_points[next_i+1]):
      # check to ensure there are no adjacent outliers within the next qualification_length number of points
      j = next_i
      while j < next_i + qualification_length:
        if j+1 == len(track_points):
          print('length {}'.format(len(track_points)))
          raise IndexError('No qualifying length of {} points with proximity threshold {}m exists!'.format(qualification_length, PROXIMITY_THRESHOLD))        
        # if outlier detected, break out of loop
        if is_outlier(track_points[j], track_points[j+1]):
          break
        j += 1
      # if all qualification_length number of points does not contain outliers
      if j == next_i + qualification_length:
        track_points[start:next_i] = [False] * (next_i - start) # mark coordinates
        break
      # else resume scanning at j
      else:
        next_i = j
    else:
      next_i += 1

def mark_noisy_coordinates(track_points):
  end = mark_null_coordinates(track_points) # Mark intial null coordinates
  mark_outliers(end, track_points)          # Mark initial outliers

def is_outlier(trkpt1, trkpt2):
  """ Returns True if the two points are beyond distance threshold """
  distance = GEOD.Inverse(trkpt1.latitude, trkpt1.longitude,
    trkpt2.latitude, trkpt2.longitude)['s12'] # distance between points in meters
  return distance > PROXIMITY_THRESHOLD

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
  #mark_noisy_coordinates(track_points)
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

def plot_actions(pkl_dir, gpx_dir, output_dir):
  '''Outputs a csv file of the detections and its corresponding gps coordinate'''

  #uses the files in the pkl directory to generate a list of filenames that is extension free
  filename_list = get_filenames_from_dir(pkl_dir)

  for filename in filename_list:
    '''Process each pkl and its corresponding gpx.'''
      
    #sets the filepaths for the required files
    pkl_file_path = set_filename_ext(pkl_dir,filename,"pkl")
    gpx_file_path = set_filename_ext(gpx_dir,filename,"gpx")
    
    video_detections = load_pickle(pkl_file_path)
    frame_count = len(video_detections)

    coordinates = process_gpx(gpx_file_path)
    '''[lat,long,timestamp]'''

    out= []

    for frame in range(0,frame_count-1):
      
      if video_detections[frame]: # appends location only if there is action detection.
        detection_location = get_corresponding_location(frame, frame_count, coordinates)
        out.append([detection_location[0],detection_location[1],video_detections[frame]])
      #[1.3025233, 103.9192801, {'run/jog': 0, 'sit': 0, 'stand': 0, 'walk': 0, 'ride': 0}]
    print(out)
      






    


def main():
    parser = argparse.ArgumentParser(description="Action plotter")
    parser.add_argument('--input', '-i', type=str, help='Input directory the pkl,gpx directory is held.')
    args = parser.parse_args()

    root_dir = args.input
    pkl_dir = os.path.join(root_dir,"pkl")
    gpx_dir = os.path.join(root_dir,"gpx")
    csv_dir = os.path.join(root_dir,"csv") #output dir

    plot_actions(pkl_dir, gpx_dir,csv_dir)


if __name__ == '__main__':
  main()