import os 
import sys 
import argparse

path = os.getcwd()
sys.path.append(path)
sys.path.append("..")

from src.live_detection import PoseRun

parser = argparse.ArgumentParser(description="Different configurations to run the model for live yoga pose detection")

parser.add_argument(
    "--task",
    dest="task",
    type=str,
    help="Whether to use cam or use from a video file",
    default="cam",
    required=False 
    )

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
  "--hide_bg",
  dest="hide_bg",
  type=bool,
  default=None,
  help="Whether to hide background or not."
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
if args.task == 'cam':
    poserun.run_pose_on_webcam(
        cam_num = args.cam, 
        black_blackground = args.hide_bg)
else:
    poserun.run_pose_on_video_source(
        source_file_path = args.vid_name, 
        model = args.model, 
        black_blackground = args.hide_bg)
