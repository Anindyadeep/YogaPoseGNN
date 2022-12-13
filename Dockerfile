# import the ubuntu latest image 

FROM ubuntu:latest 
FROM python:3.8 

# Install the initial apt packages 

RUN pip install --upgrade pip
RUN python3 -m venv /opt/venv
RUN . /opt/venv/bin/activate
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

# Create the folder directories inside the container.

RUN mkdir YogaPoseGNN
RUN mkdir YogaPoseGNN/src 
RUN mkdir YogaPoseGNN/saved_models
RUN mkdir YogaPoseGNN/Models 
RUN mkdir YogaPoseGNN/Data 

# COPY all the required files to those directories.

COPY streamlit_app.py YogaPoseGNN
COPY Data/ YogaPoseGNN/Data/
COPY Models/ YogaPoseGNN/Models/ 
COPY saved_models/ YogaPoseGNN/saved_models/
COPY run.py YogaPoseGNN 
COPY src/ YogaPoseGNN/src  
COPY favicon.ico YogaPoseGNN
COPY README.md YogaPoseGNN
COPY me.md YogaPoseGNN

# Install all the required python libraries.

RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu
RUN pip3 install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
RUN pip3 install mediapipe streamlit opencv-python 

# Set the workdir and run the commmand

WORKDIR "/YogaPoseGNN"
RUN ls 
EXPOSE 8900
ENTRYPOINT ["streamlit", "run"]
CMD ["streamlit_app.py"]