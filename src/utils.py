import os
import random
import shutil
import sys
import warnings
from pathlib import Path
from typing import Optional

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

mp_drawing = mp.solutions.drawing_utils  # type: ignore
mp_drawing_styles = mp.solutions.drawing_styles  # type: ignore
mp_pose = mp.solutions.pose  # type: ignore
warnings.filterwarnings("ignore")

BASEDIR = Path(__file__).resolve(strict=True).parent.parent
sys.path.append(str(BASEDIR))
sys.path.append("..")


class PoseUtils(object):
    def __init__(
        self,
        parent_path: Optional[str] = None,
        train_image_folder_name: Optional[str] = None,
        test_image_folder_name: Optional[str] = None,
    ):
        """
        args:
        -----
        parent_path : (str) the root path of the project
        train_image_folder_name : (str) the name of the train image folders
        test_image_folder_name : (str) the name of the test image folders
        """
        self.base_path = BASEDIR if parent_path is None else parent_path
        self.train_image_folder_name = (
            "DATASET/TRAIN" if train_image_folder_name is None else train_image_folder_name
        )
        self.test_image_folder_name = (
            "DATASET/TEST" if test_image_folder_name is None else test_image_folder_name
        )

    def random_shift_data_points_from_test_to_train_folder(
        self, train_folder_name: str, test_folder_name: str
    ):
        """
        This function will randomly take some datapoints from the test image folder and populate the train folder
        """
        train_folder = os.path.join(self.base_path, self.train_image_folder_name, train_folder_name)
        test_folder = os.path.join(self.base_path, self.test_image_folder_name, test_folder_name)

        test_images = os.listdir(test_folder)
        sample_length = (
            int(len(test_images) * 0.5) if len(test_images) > 200 else int(0.3 * len(test_images))
        )
        sample_test_images = random.sample(test_images, sample_length)

        for sample_test_img in tqdm(sample_test_images, total=sample_length):
            source_path = os.path.join(test_folder, sample_test_img)
            target_path = train_folder
            shutil.move(source_path, target_path)
        print(
            f"finished moving {sample_length} images out of {len(test_images)} from {test_folder_name} to {train_folder}"
        )

    def get_label_dict(self):
        """
        Generating the labels from the image folders
        """
        path = os.path.join(self.base_path, self.train_image_folder_name)
        labels = os.listdir(path)
        num_labels = np.arange(len(labels))
        zip_iterator = zip(labels, num_labels)
        return dict(zip_iterator)

    def get_pose_landmark_positions(self, pose_landmarks):
        """
        Picking the pose landmarks co-ordinates from mediapipe's pose landmark skeleton graph.
        """

        left_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]  # 11
        right_shoulder = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]  # 12

        left_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW]  # 13
        right_elbow = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW]  # 14

        left_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST]  # 15
        right_wrist = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST]  # 16

        left_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]  # 23
        right_hip = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP]  # 24

        left_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE]  # 25
        right_knee = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE]  # 26

        left_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE]  # 27
        right_ankle = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE]  # 28

        left_foot_index = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]  # 31
        right_foot_index = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX]  # 32

        pose_landmarks_coords = {
            "11: left_shoulder_x": left_shoulder.x,
            "11: left_shoulder_y": left_shoulder.y,
            "11: left_shoulder_z": left_shoulder.z,
            "12: right_shoulder_x": right_shoulder.x,
            "12: right_shoulder_y": right_shoulder.y,
            "12: right_shoulder_z": right_shoulder.z,
            "13: left_elbow_x": left_elbow.x,
            "13: left_elbow_y": left_elbow.y,
            "13: left_elbow_z": left_elbow.z,
            "14: right_elbow_x": right_elbow.x,
            "14: right_elbow_y": right_elbow.y,
            "14: right_elbow_z": right_elbow.z,
            "15: left_wrist_x": left_wrist.x,
            "15: left_wrist_y": left_wrist.y,
            "15: left_wrist_z": left_wrist.z,
            "16: right_wrist_x": right_wrist.x,
            "16: right_wrist_y": right_wrist.y,
            "16: right_wrist_z": right_wrist.z,
            "23: left_hip_x": left_hip.x,
            "23: left_hip_y": left_hip.y,
            "23: left_hip_z": left_hip.z,
            "24: right_hip_x": right_hip.x,
            "24: right_hip_y": right_hip.y,
            "24: right_hip_z": right_hip.z,
            "25: left_knee_x": left_knee.x,
            "25: left_knee_y": left_knee.y,
            "25: left_knee_z": left_knee.z,
            "26: right_knee_x": right_knee.x,
            "26: right_knee_y": right_knee.y,
            "26: right_knee_z": right_knee.z,
            "27: left_ankle_x": left_ankle.x,
            "27: left_ankle_y": left_ankle.y,
            "27: left_ankle_z": left_ankle.z,
            "28: right_ankle_x": right_ankle.x,
            "28: right_ankle_y": right_ankle.y,
            "28: right_ankle_z": right_ankle.z,
            "31: left_foot_index_x": left_foot_index.x,
            "31: left_foot_index_y": left_foot_index.y,
            "31: left_foot_index_z": left_foot_index.z,
            "32: right_foot_index_x": right_foot_index.x,
            "32: right_foot_index_y": right_foot_index.y,
            "32: right_foot_index_z": right_foot_index.z,
        }

        return pose_landmarks_coords

    def get_pose_marks_coords_dict_for_image(self, img_path: str, pose_label: str, pose_landmarks):
        """
        args
        ----
        img_path : the path of the image
        pose_label : the label of the image
        pose_landmarks : the pose landmarks generated from mediapipe pose model

        returns
        -------
        A dict of co-ordinate information of different pose joints.

        """
        pose_land_marks_coords = self.get_pose_landmark_positions(pose_landmarks)
        pose_land_marks_coords["Image Path"] = img_path
        pose_land_marks_coords["Label"] = pose_label

        return pose_land_marks_coords

    def read_single_image(self, file: str):
        """
        Reads a single image file and extracts the corresponding pose landmarks
        args:
        ----
        file : the image file

        returns:
        --------
        mediapipe pose landmark results
        """

        BG_COLOR = (192, 192, 192)
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
        ) as pose:
            image = cv2.imread(file)
            results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return results

    def create_landmark_csv_from_single_image_folder(self, image_folder_name: str, label: str):
        """
        Iterates over each image files from the folder and extract the keypoints and create a csv file. This is only for a single image folder.
        args:
        ----
        image_folder_name : the name of the target image folder to convert
        label : the label or name of the class

        returns
        -------
        A pandas dataframe object
        """

        image_folder_path = os.path.join(self.base_path, image_folder_name)
        images = os.listdir(image_folder_path)
        img_list = []

        for img in tqdm(images, total=len(images)):
            img_path = os.path.join(image_folder_path, img)
            img_results = self.read_single_image(img_path)

            if img_results.pose_landmarks is not None:
                img_list.append(
                    self.get_pose_marks_coords_dict_for_image(
                        img_path, label, img_results.pose_landmarks
                    )
                )

        df = pd.DataFrame.from_dict(img_list)  # type: ignore
        if "Unnamed: 0" in df.columns:
            df = df.drop(["Unnamed: 0"], axis=1)
        df = df.reset_index(drop=True)
        return df

    def create_csv_from_landmarks(self, save_csv_folder_name: str, test: Optional[bool] = False):
        """
        Creates a csv file by iterating all the class image folders and creating the csv containing pose keypoints
        The generated CSV file will be saved in Data/CSV

        args:
        ----
        save_csv_folder_name : (str) the name of the csv file to save
        test : (boolean) whether to save in train folder or test folder.
        """
        train_base_path = os.path.join(self.base_path, self.train_image_folder_name)
        train_csvs_path = os.path.join(self.base_path, save_csv_folder_name)
        labels = os.listdir(train_base_path)
        label_dict = self.get_label_dict()

        if len(os.listdir(train_csvs_path)) != 0:
            print("Files are already present")

        else:
            print("Processing started ...")
            for i, label in enumerate(labels):
                class_label = label_dict[label]
                folder_name = label
                temp_image_folder = os.path.join(self.train_image_folder_name, label)
                single_csv_data = self.create_landmark_csv_from_single_image_folder(
                    temp_image_folder, class_label
                )

                if "Unnamed: 0" in single_csv_data.columns:
                    single_csv_data = single_csv_data.drop(["Unnamed: 0"], axis=1)

                single_csv_data = single_csv_data.reset_index(drop=True)
                print(f"completed {label}.csv with shape: {single_csv_data.shape}")

                csv_save_path_name = (
                    os.path.join(train_csvs_path, f"{label}_train.csv")
                    if test == False
                    else os.path.join(train_csvs_path, f"{label}_test.csv")
                )
                single_csv_data.to_csv(csv_save_path_name)
            print("Training data making completed ....")

    def concat_all_csv_into_one(
        self,
        all_csv_path: str,
        save_file_folder_name: str,
        shuffle: Optional[bool] = True,
        split: Optional[float] = None,
        test: Optional[bool] = False,
    ):
        """
        Concats all the CSV files in a given folder and saves the final concated csv file to an another folder
        args:
        ----
        all_csv_path : (str) the CSV folder path (either train or test)
        save_csv_folder_name : (str) the name of the folder to save the concated csv files
        shuffle : (boolean) Whether to shuffle rows or not.
        split : (float) Splits the concated data into two parts (e.g. train and validation)
        test : (boolean) Whether the file is a test file or not

        """
        all_csv_file_paths = os.path.join(self.base_path, all_csv_path)
        save_file_folder_path = os.path.join(self.base_path, save_file_folder_name)

        if ".csv" not in os.listdir(
            save_file_folder_path
        ):  # change this later to make it more general
            print("Concatination process started ...")
            concated_csv_data = pd.concat(  # must be for those woth train or test
                pd.read_csv(os.path.join(all_csv_file_paths, file))
                for file in tqdm(
                    os.listdir(all_csv_file_paths), total=len(os.listdir(all_csv_file_paths))
                )
            )

            if "Unnamed: 0" in concated_csv_data.columns:
                concated_csv_data = concated_csv_data.drop(["Unnamed: 0"], axis=1)

            if shuffle:
                concated_csv_data = concated_csv_data.sample(frac=1)

            if split is not None and split < 0.5:
                concated_csv_data_train, concated_csv_data_valid = self.train_validation_split(
                    concated_csv_data, split
                )
                concated_csv_data_train = concated_csv_data_train.reset_index(drop=True)
                concated_csv_data_valid = concated_csv_data_valid.reset_index(drop=True)

                if test == False:
                    concated_csv_data_train.to_csv(
                        os.path.join(save_file_folder_path, "train_data.csv")
                    )
                    concated_csv_data_valid.to_csv(
                        os.path.join(save_file_folder_path, "valid_data.csv")
                    )
                else:
                    concated_csv_data_train.to_csv(
                        os.path.join(save_file_folder_path, "test_data.csv")
                    )

            else:
                if test == False:
                    concated_csv_data = concated_csv_data.reset_index(drop=True)
                    concated_csv_data.to_csv(os.path.join(save_file_folder_path, "train_data.csv"))
                else:
                    concated_csv_data.to_csv(os.path.join(save_file_folder_path, "test_data.csv"))
            print("Done!")
        else:
            print("Files are present already")

    def train_validation_split(self, df: pd.DataFrame, split_size: float):
        """
        Splits the file into train and validation files.
        args:
        ----
        df : (int) the dataframe to split
        split_size : (float) the train : validation ratio
        """
        total_size = df.shape[0]
        valid_size = int(total_size * split_size)
        train_size = total_size - valid_size
        train_df = df.iloc[:train_size, :]
        valid_df = df.iloc[train_size:, :]
        return train_df, valid_df


if __name__ == "__main__":
    # pose_dir_names = [
    #     "downdog",
    #     "goddess",
    #     "plank",
    #     "tree",
    #     "warrior2"
    # ]

    pos_util = PoseUtils()
    # for pose in pose_dir_names:
    #     pos_util.random_shift_data_points_from_test_to_train_folder(pose, pose)

    pos_util.create_csv_from_landmarks("Data/CSVs/train", test=False)
    pos_util.concat_all_csv_into_one("Data/CSVs/train", "Data/raw", shuffle=True, test=False)

    pos_util.create_csv_from_landmarks("Data/CSVs/test", test=True)
    pos_util.concat_all_csv_into_one("Data/CSVs/test", "Data/raw", shuffle=True, test=True)
