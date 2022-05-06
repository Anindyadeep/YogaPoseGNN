import os 
import sys 
import cv2
import torch
import warnings
import numpy as np
import mediapipe as mp 
import streamlit as st 
from pathlib import Path

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(BASEDIR)
sys.path.append("..")


from src.utils import PoseUtils
from src.dataset import YogaPosDataset
from src.train import TrainModel
from Models.base_gnn_model import Model 

class PoseRun(object):
  def __init__(self, base_path = None, device = None):
    self.base_path = BASEDIR if base_path is None else base_path
    self.LABELS = {
            0 : "downdog",
            1 : "goddess",
            2 : "plank",
            3 : "tree",
            4 : "warrior"
        }

    self._font = cv2.FONT_HERSHEY_SIMPLEX
    self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
    self.pose_util = PoseUtils()
    
  def load_model(self, model = None, model_path = None):
    if model is None and model_path is None:
      model = Model(3, 64, 32, 5).to(self.device)
      model_path = os.path.join(self.base_path, "saved_models/base_model.pth")
      model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

    else:
      model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))
    return model 
  
  
  def load_edge_index(self):
    edge_index = YogaPosDataset(os.path.join(self.base_path, "Data/"), "train_data.csv").edge_index.to(self.device)
    return edge_index
  
  def _provide_video_writer_config(self, cap, save_as = None):
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
      
    size = (frame_width, frame_height)
    vid_result = cv2.VideoWriter(os.path.join(BASEDIR, f'Video_results/result.avi') if save_as is None else os.path.join(BASEDIR, f'Video_results/{save_as}'), 
                             cv2.VideoWriter_fourcc(*'MJPG'),
                             30, size)
    return vid_result


  def run_detection_on_app(self, cam_num = None):
        self.model = self.load_model() 
        edge_index = self.load_edge_index()

        col1, col2 = st.columns([.33,1])
        run = col1.button('Switch on the video')
        stop = col2.button('Switch off the video')

        FRAME_WINDOW = st.image([])
        cap = cv2.VideoCapture(0 if cam_num is None else cam_num)
        with mp_pose.Pose(min_detection_confidence = 0.5, min_tracking_confidence = 0.5) as pose:
          while run and not stop:
                sucess, frame = cap.read()
                if not sucess:
                      continue 

                image = cv2.flip(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), 1)
                results = pose.process(image)
                if not results.pose_landmarks: 
                  continue

                pos_dict = self.pose_util.get_pose_landmark_positions(results.pose_landmarks)
                x = torch.tensor(np.array(list(pos_dict.values())).reshape(14, 3), dtype=torch.float32).to(self.device)
                out = self.model(x, edge_index, torch.tensor([0]))
                
                predictions = out.argmax(dim=1).item()
                mp_drawing.draw_landmarks(
                  image,
                  results.pose_landmarks,
                  mp_pose.POSE_CONNECTIONS,
                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
                
                image = cv2.flip(image, 1)

                cv2.putText(
                  image, 
                  f"Position: {self.LABELS[predictions]}", 
                  (50, 50), 
                  self._font, 1.5, 
                  (0, 255, 255), 
                  4, 
                  cv2.LINE_4)

                FRAME_WINDOW.image(image)
          else:
                pass 


  
  def run_video(self, video_name = None, cam_num = None, model = None, capture_save_as = None):
    self.model = self.load_model() if model is None else model 
    edge_index = self.load_edge_index()

    cap = cv2.VideoCapture((0 if cam_num is None else cam_num) if video_name is None else str(os.path.join(self.base_path, f"Sample_video/{video_name}")))
    vid_result = self._provide_video_writer_config(cap, save_as = capture_save_as) if capture_save_as else None 

    with mp_pose.Pose(
      min_detection_confidence = 0.5,
      min_tracking_confidence = 0.5) as pose:
      while cap.isOpened():
        success, image = cap.read()
        if not success:
          print("Ignoring empty camera frame")
          continue 

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = pose.process(image)

        if not results.pose_landmarks:
          continue 

        pos_dict = self.pose_util.get_pose_landmark_positions(results.pose_landmarks)
        x = torch.tensor(np.array(list(pos_dict.values())).reshape(14, 3), dtype=torch.float32).to(self.device)
        out = self.model(x, edge_index, torch.tensor([0]))
        predictions = out.argmax(dim=1).item()

        cv2.putText(image, 
                f"Position: {self.LABELS[predictions]}", 
                (50, 50), 
                self._font, 1.5, 
                (0, 255, 255), 
                4, 
                cv2.LINE_4)
        
        image.flags.writeable = True
        image = cv2.flip(image, 1)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
          image,
          results.pose_landmarks,
          mp_pose.POSE_CONNECTIONS,
          landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        
        cv2.imshow('Frame', cv2.flip(image, 1))
        if capture_save_as is not None:
          vid_result.write(cv2.flip(image, 1))
          
        if cv2.waitKey(5) & 0xFF == 27:
          break
    cap.release()
    if capture_save_as:
      vid_result.release() 


if __name__ == '__main__':
  pose_run = PoseRun()
  pose_run.run_video(video_name="")
  