import warnings

import mediapipe as mp
import torch
import uvicorn
from fastapi import FastAPI
from fastapi.logger import logger
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore")

from src.live_detection import PoseRun

from datamodels import FrameParameters, ResponseModel, StreamedData
from vidstream import VideoCam

########################### Major changes to make ######################
"""
TODO:

1. Change from streamed image input to image results format 
2. We do not need much load by providing byte image as it's less efficient 
"""
########################################################################


vid = VideoCam()

app = FastAPI(debug=True)
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


device = "cuda" if torch.cuda.is_available() else "cpu"


@app.on_event("startup")
async def startup_event():
    """
    Initializes different FastAPI and its variables
    """
    logger.info(f"Pytorch is using device: {device}")
    poserun = PoseRun(device=device)
    model = poserun.load_model()

    app.package = {"model": model, "poserun": poserun}  # type: ignore


@app.get("/")
async def about():
    return {
        "message": "Welcome to YogaPose FastAPI",
        "direction": "Head over to /predict to get predictions of your frames",
    }


@app.post("/predict", response_model=ResponseModel)
async def predict_streamed_data(request: FrameParameters, body: StreamedData):
    """Predict endpoint POST request

    Args:
        request (FrameParameters): This is of type FrameParameters BaseModel and should contain infor about image height, width, batch size
        body (StreamedData): This should contain a  string which represents the bytes formatted of streamed frames
    """
    flattened_decoded_frames = vid.decode_str_to_batched_frames(body.streamed_byte_data)
    decoded_frames = flattened_decoded_frames.reshape(
        request.batch_size, request.image_width, request.image_height, 3
    )

    model = app.package["model"]  # type: ignore

    with torch.no_grad():
        pass
