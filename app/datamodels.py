from pydantic import BaseModel


class FrameParameters(BaseModel):
    """
    frame_number : (int) The number of frame passed through
    batch_size : (int) The intended batch size
    image_width : (int) The width of the image
    image_height : (int) The height of the image
    """

    frame_number: int
    batch_size: int
    image_width: int
    image_height: int


class StreamedData(BaseModel):
    """
    streamed_byte_data : (str) Contains a string representation of byte format for frames
    """

    streamed_byte_data: str


class ResponseModel(BaseModel):
    """
    frame_number : (int) The number of frame passed through
    batch_size : (int) The intended batch size
    predicted_yoga_pose : (str) The predicted yoga pose class
    prediction_confidence : (float) The confidence of the prediction
    """

    frame_number: int
    batch_size: int
    predicted_yoga_pose: str
    prediction_confidence: float
