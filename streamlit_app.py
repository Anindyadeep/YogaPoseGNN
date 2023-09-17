import os
import sys
import tempfile
import warnings
from typing import Optional, Union

import cv2
import streamlit as st
from matplotlib.pyplot import show

warnings.filterwarnings("ignore")
path = os.getcwd()
sys.path.append(path)
sys.path.append("..")

from src.live_detection import PoseRun


class StreamlitApp(object):
    def __init__(self, device: Optional[str] = "cpu"):
        """
        This app supports CPU in default when it comes to deployement through docker.
        The support for GPU will be provided in the coming version.
        """
        self.device = device
        self.poserun = PoseRun(device=self.device)

    def about_project(self):
        st.title("Real time Yoga Pose APP")
        st.text("Streamlit 💚 Mediapipe 💚 PyTorch Geometric")

        st.markdown("## Welcome to YogaPoseGNN :smile:")
        gif_path = os.path.join(path, "Images/tree_PtFptRsa.gif")
        st.markdown("![Alt Text](https://media.giphy.com/media/CocmEsPoERVdDj4PS5/giphy.gif)")

        with open("README.md") as f:
            contents = f.read()
        contents = contents[65:]
        st.markdown(contents)

    def demo_on_sample_video(self):
        video_file = st.file_uploader("video")
        video_temp_file = tempfile.NamedTemporaryFile(delete=False)
        if video_temp_file is not None and video_file is not None:
            video_temp_file.write(video_file.read())

        show_background = st.checkbox("Hide background")
        cap = cv2.VideoCapture(video_temp_file.name)
        app_window = st.image([])

        if cap.isOpened():
            for generated_frame in self.poserun.generate_results(
                cap=cap, black_blackground=show_background
            ):
                app_window.image(cv2.cvtColor(generated_frame, cv2.COLOR_RGB2BGR))  # type: ignore
                cv2.waitKey(10)
        else:
            pass

    def demo_on_webcam(self, cam_source: Optional[Union[str, int]] = None):
        st.write("We are sorry, the webcam feature will not work if not run on localhost")

        col1, col2, col3 = st.columns([1, 1, 1])
        cam_on = col1.button("Switch on the Video")
        cam_stop = col2.button("Switch off the video")
        show_background = col3.checkbox("Hide background")

        cam_source = -1 if cam_source is None else cam_source
        cap = cv2.VideoCapture(cam_source)
        app_window = st.image([])

        while cam_on and not cam_stop:
            for generated_frame in self.poserun.generate_results(
                cap=cap, black_blackground=show_background
            ):
                app_window.image(cv2.cvtColor(generated_frame, cv2.COLOR_RGB2BGR))
                cv2.waitKey(10)
        else:
            pass

    def about_me(self):
        st.title("Real time Yoga Pose APP")
        st.text("Streamlit 💚 Mediapipe 💚 PyTorch Geometric")
        with open("me.md") as f:
            contents = f.read()
        st.markdown(contents)

    def webapp(self):
        activities = ["About the project", "Demo", "About me"]
        choice = st.sidebar.selectbox("Select", activities)

        if choice == "About the project":
            self.about_project()

        elif choice == "Demo":
            st.title("Real time Yoga Pose APP")
            st.text("Streamlit 💚 Mediapipe 💚 PyTorch Geometric")
            type_choice = st.selectbox("Select from the options", ["From file", "From webcam"])
            if type_choice == "From file":
                self.demo_on_sample_video()
            else:
                self.demo_on_webcam()
        else:
            self.about_me()


if __name__ == "__main__":
    app = StreamlitApp().webapp()
