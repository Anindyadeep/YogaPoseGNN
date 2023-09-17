# This program converts continuous stream of frames into stream of bytes
# so that it can be used as a POST request for APIs

import warnings
from typing import Generator, List, Union

import cv2
import numpy as np

warnings.filterwarnings("ignore")


class VideoCam:
    def __init__(self):
        pass

    def batched_frame_encode(self, batched_frames: List[np.ndarray]) -> str:
        """Encodes a list of numpy array into byte format -> string

        Args:
            batched_frames (List[np.ndarray]): List of numpy array represernting image frames

        Returns:
            str: String containing the byte information of the array
        """
        batched_frames_np = np.stack(batched_frames).reshape(-1)
        to_bytes = batched_frames_np.tobytes()
        bytes_to_str = to_bytes.decode("ISO-8859-1")
        return bytes_to_str

    def decode_str_to_batched_frames(self, encoded_frames: str) -> np.ndarray:
        """Decodes the string representation back to batches of numpy array frames

        Args:
            encoded_frames (str): String representation of the frames

        Returns:
            _type_: (numpy.ndarray) numpy array represernting image frames
        """
        encoded_frames_bytes = encoded_frames.encode("ISO-8859-1")
        encoded_frames_np = np.frombuffer(encoded_frames_bytes, dtype=np.uint8)
        return encoded_frames_np

    def capture_video(self, source: Union[int, str] = 0, batch_size: int = 16) -> Generator:
        """Captures video frames from a given source

        Args:
            source (Union[int, str], optional): The source of video capture. Defaults to 0.
            batch_size (int, optional): Batch size. Defaults to 16.

        Yields:
            Generator: Generates string representation of the byte format of video frames
        """
        frame_count = 0
        batched_frames = []

        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            image = np.array(frame)
            frame_count += 1
            batched_frames.append(image)
            cv2.imshow("Frame", frame)

            if frame_count == batch_size:
                encoded_batched_frames = self.batched_frame_encode(batched_frames=batched_frames)
                yield encoded_batched_frames

                frame_count = 0
                batched_frames = []

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()
