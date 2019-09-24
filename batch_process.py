import os.path 
import os
import argparse
import sys

def main(folder_path):

    #to implement validity check of the directory structure
    #Create folder if does not exist  
    '''Directory should contain'''
    '''csv, csv_conv, gpx, log, pkl, video_src, video_vis'''

    video_src_path = os.path.join(folder_path,"video_src")
    pkl_folder_path = os.path.join(folder_path,"pkl")
    sys.argv = [sys.argv[0]]#clears all sys arguments

    #1.Extract telemetries, note that GoPro video that are processed with a video editor have the telemetries and metadata stripped.
    import extract_telemetries as ET

    print("Extracting telemetries...")
    ET.main(video_src_path)
    print("Telemetries extracted.")

    #2.Detect Actions
    import detect_actions_pcn as DA

    print("Detecting actions...")
    DA.main(video_src_path)
    print("Action detection completed.") 

    #3.Plot Actions
    import plot_actions_pcn as PA 

    print("Plotting actions to csv...")
    PA.main(pkl_folder_path)
    print("Plot actions completed.")

if __name__ == '__main__':
  main(sys.argv[1])