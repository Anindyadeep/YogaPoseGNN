import os
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
import torch
from Models.base_gnn_model import Model

from src.dataset import YogaPosDataset
from src.train import TrainModel
from src.utils import PoseUtils

mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
mp_pose = mp.solutions.pose  # type: ignore
warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(BASEDIR))
sys.path.append("..")


class PoseRun(object):
    def __init__(self, base_path: Optional[str] = None, device: Optional[str] = None):
        """
        args:
        -----
        base_path : (str) The root path of the project
        device : (str) The device to run the model
        """

        self.base_path = BASEDIR if base_path is None else base_path
        self.LABELS = {0: "downdog", 1: "goddess", 2: "plank", 3: "tree", 4: "warrior"}

        self._font = cv2.FONT_HERSHEY_SIMPLEX
        self.device = ("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.pose_util = PoseUtils()

    def load_model(self, model: Optional[torch.nn.Module] = None, model_path: Optional[str] = None):
        """
        Loads the model PyTorch model for pretrained weights
        args:
        -----
        model : PyTorch (torch.nn.Module) object
        model_path : The weights path (.pth) file path

        returns:
        --------
        The same (torch.nn.Module) object model loaded with pretrained weights
        """

        if model is None and model_path is None:
            model = Model(3, 64, 32, 5).to(self.device)
            model_path = os.path.join(self.base_path, "saved_models/base_model.pth")
            model.load_state_dict(torch.load(model_path, map_location=torch.device(self.device)))

        else:
            model.load_state_dict( #type: ignore
                torch.load(model_path, map_location=torch.device(self.device))  # type: ignore
            )  # type: ignore
        return model

    def load_edge_index(self):
        """
        Loads the edge indices from the pose graph edges from mediapipe
        """
        edge_index = YogaPosDataset(
            os.path.join(self.base_path, "Data/"), "train_data.csv"
        ).edge_index.to(self.device)
        return edge_index

    def _provide_video_writer_config(self, cap, save_as: Optional[str] = None):
        """
        Provides the required configuraton to save the video
        args:
        -----
        cap : The capture object
        save_as : The file name to save the generated video file with corresponding predictions.
        """
        frame_width = int(cap.get(3))
        frame_height = int(cap.get(4))

        size = (frame_width, frame_height)
        file_to_save_as = (
            os.path.join(BASEDIR, f"Video_results/result.avi")
            if save_as is None
            else os.path.join(BASEDIR, f"Video_results/{save_as}")
        )
        vid_result = cv2.VideoWriter(file_to_save_as, cv2.VideoWriter_fourcc(*"MJPG"), 30, size)
        return vid_result

    def generate_results(
        self,
        cap: cv2.VideoCapture,
        model: Optional[torch.nn.Module] = None,
        black_blackground: Optional[bool] = False,
    ):
        """
        yields the frame which is the prediction from the GNN model
        args:
        -----
        cap : cv2.VideoCapture() object
        model : (torch.nn.Module) PyTorch model for the prediction

        yields:
        -------
        A NumPy array containing the model prediction results
        """
        self.model = self.load_model() if model is None else model
        edge_index = self.load_edge_index()

        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            while cap.isOpened():
                sucess, image = cap.read()
                if not sucess:
                    print("Ignoring empty camera frame")
                    break

                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = pose.process(image)

                if not results.pose_landmarks:
                    continue

                pos_dict = self.pose_util.get_pose_landmark_positions(results.pose_landmarks)
                x = torch.tensor(
                    np.array(list(pos_dict.values())).reshape(14, 3), dtype=torch.float32
                ).to(self.device)
                out = self.model(x, edge_index, torch.tensor([0]))  # type: ignore
                predictions = out.argmax(dim=1).item()

                canvas = np.zeros_like(image) if black_blackground else image
                cv2.putText(
                    canvas,
                    f"Position: {self.LABELS[predictions]}",
                    (50, 50),
                    self._font,
                    1.5,
                    (0, 255, 255),
                    4,
                    cv2.LINE_4,
                )

                canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

                mp_drawing.draw_landmarks(
                    canvas,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

                yield canvas

    def run_pose_on_webcam(
        self,
        cam_num: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        black_blackground: Optional[bool] = False,
    ):
        """
        Run the yoga pose prediction on web cam.
        args:
        -----
        cam_num : (int) the camera number from the device
        model : (torch.nn.Module) PyTorch model for the prediction
        black_blackground : Optional[bool] Turns the background to black if True
        """
        cam_num = -1 if cam_num is not None else cam_num
        cap = cv2.VideoCapture(cam_num)

        for generated_frame in self.generate_results(
            cap=cap, model=model, black_blackground=black_blackground
        ):
            cv2.imshow("Frame", generated_frame)
            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()

    def run_pose_on_video_source(
        self,
        source_file_path: str,
        model: Optional[torch.nn.Module] = None,
        black_blackground: Optional[bool] = False,
        save_capture_as: Optional[bool] = False,
    ):
        """
        Runs the model on a video source.
        args:
        ----
        source_file : The path of the video
        model : (torch.nn.Module) The PyTorch model
        black_blackground : (boolean) Whether to hide the visual background and only show the pose.
        save_capture_as : (boolean) Whether to save the results or not
        """

        cap = cv2.VideoCapture(source_file_path)
        save_capture_file_name = str(source_file_path.split("/")[-1]) + ".avi"
        video_result = (
            self._provide_video_writer_config(cap, save_as=save_capture_file_name)
            if save_capture_as
            else None
        )

        for generated_frame in self.generate_results(
            cap=cap, model=model, black_blackground=black_blackground
        ):
            cv2.imshow("Frame", generated_frame)
            if save_capture_as:
                video_result.write(generated_frame)  # type: ignore

            if cv2.waitKey(10) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
        if save_capture_as:
            video_result.release()  # type: ignore
            print(f"saved as: {save_capture_file_name}")
