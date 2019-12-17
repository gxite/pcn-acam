import subprocess
import argparse
import tqdm
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'submodules/gopro')))
#from library import utils
from submodules.gopro.gopro_telemetry import GoProTelemetry

#utils.ensure_python_version(3)

def extract_telemetry_single(video_path, reprocess):
  """Extract telemetry for a single GoPro capture"""
  gopro_telemetry = GoProTelemetry(video_path, reprocess=reprocess, prepend_filename_with_serial=False, append_filename_with_timestamp=False)
  #gopro_telemetry.extract_all()
  gopro_telemetry.extract_gpx()

def extract_telemetry_all(dir_path, reprocess):
  """Extract telemetry for all GoPro captures in directory"""
  extraction_errors = []
  gopro_videos = list(filter(lambda f: f.endswith('.MP4'), os.listdir(dir_path)))
  for filename in tqdm.tqdm(gopro_videos, total=len(gopro_videos), desc='Processing ' + dir_path):
    video_path = os.path.abspath(os.path.join(dir_path, filename))
    try:
      extract_telemetry_single(video_path, reprocess)
    except subprocess.CalledProcessError:
      extraction_errors.append(video_path)
  
  if extraction_errors:
    print()
    print('Telemetry extraction failed for the following captures:')
    print('\n'.join(extraction_errors))
    print()

def main(video_src_path=""):
  parser = argparse.ArgumentParser(description="Telemetry extractor wrapper for GoPro captures")
  parser.add_argument('--reprocess', '-r', action='store_true', help='Flag to reprocess telemetry')
  parser.add_argument('--video_path', '-v', type=str, help='Video path for GoPro capture')
  parser.add_argument('--folder_path', '-f', type=str, help='Directory path for GoPro captures')
  args = parser.parse_args()

  if video_src_path != "":
    args.folder_path = video_src_path

  if args.video_path:
    try:
      extract_telemetry_single(os.path.abspath(args.video_path), args.reprocess)
    except subprocess.CalledProcessError as e:
      print('Telemetry extraction for {} has problems: {}'.format(args.single, str(e)))
  elif args.folder_path:
    extract_telemetry_all(os.path.abspath(args.folder_path), args.reprocess)

if __name__ == '__main__':
  main()
