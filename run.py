import os 
import sys 
import argparse

path = os.getcwd()
sys.path.append(path)
sys.path.append("..")

from src.live_detection import PoseRun

parser = argparse.ArgumentParser(description="Different configurations to run the model for live yoga pose detection")

parser.add_argument(
  "--device",
  dest="device",
  type=str,
  default="cpu",
  help="Enable Cuda or not",
  required = False
)

parser.add_argument(
  "--cam",
  dest="cam",
  type=int,
  default=None,
  help="The camera number to run the model in live camera",
  required=False
)

parser.add_argument(  
  "--vid_name",
  dest="vid_name",
  type=str,
  default=None,
  help="The name of the video [video.mp4] to run the model on some video. The video must be saved in [Sample_video/] path",
  required=False
)

parser.add_argument(
  "--save_as",
  dest="save_as",
  type=str,
  default=None,
  help="Saving the video if required and it will be saved on [Video_results/] folder automatically with the specified name"
)


parser.add_argument(
  "--model",
  dest="model",
  type=str,
  default=None,
  help="The model to run this full detection process."
)

args = parser.parse_args()

poserun = PoseRun(device=args.device)
poserun.run_video(
  video_name=args.vid_name,
  cam_num=args.cam,
  capture_save_as=args.save_as,
  model=args.model
)