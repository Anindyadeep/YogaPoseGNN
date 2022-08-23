# **YogaPosGNN** `Version 2.0`

![Alt Text](Images/warrior-result_2IogI1xJ.gif)

[GIF credits](https://www.youtube.com/watch?v=k4qaVoAbeHM&ab_channel=Howcast). This is another kinda same project done and made on the top of [SignLangGNN](https://github.com/Anindyadeep/SignLangGNN). But this project is much more robust and accurate and can perform real time Yoga position classification using Graph Neural Networks. The best part of this project is that the CPU utlization. As its just using some pixel co-ordinate changes in the video. Also Graph neural networks are emerging more and more in several aspects of computer vision. So this problem is framed as a graph classification problem. I used a simple two Graph Attention layers and a softmax classifier as the network architecture. In just 20 epochs it gives an accuracy of `0.91` and `0.89` of train and test accuracy respectively. `Better than the previous versions`.

----
<br>

## **Contents**

1. [ Running the project locally ](#run_project_locally)
2. [ How to train a new model or retrain the existing model ](#re_train_model)
3. [ Results ](#results)
4. [ Future Works ](#future_works)

----
<br>

<a name="run_project_locally"></a>
## **Running the project Locally**
<br>

### **Using Docker**

The docker image is been released in **DockerHub** 🥳
```bash
$ docker run --privileged --device=/dev/video0:/dev/video0 anindyadeep/yoga_pose_gnn:master-3e72318
```

### **Cloning the repo and building the image**

You can clone the project using:
```bash
$ git clone https://github.com/Anindyadeep/YogaPosGNN.git
```
And after that if you want to run the build the docker image locally and then run the following.

```bash
$ docker build -t <image_name> .
$ docker run --privileged --device=/dev/video0:/dev/video0 <image_name>
```

Just replace the `<image name>` with the any arbitary name for e.g. `test_image`.
providing OS and camera privilages are very much important, so we need to use the additional commands, otherwise, you might not use video based results. 

----

<br>

### **Runing the project using python**

This project is using **OpenCV**, **TensorFlow**, **PyTorch**, **PyTorch Geometric**, and **mediapipe**.

First clone the project repo. Create a new environment using `conda` or `virtualenv`. Once created, you need to install the following packages. First checkout the whether you have cuda installed in your system or not. Based on that go to the **[PyTorch's](https://pytorch.org/)** website and intall PyTorch. Once done, move over to **[PyTorch Geometric's](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html)**. Be careful while installing the packages. Make sure you install the current version. 

Install opencv, tensorflow and mediapipe as follows:

```bash

$ pip install tensorflow
$ pip install mediapipe
$ pip install opencv-python

```

Once all the required libraries are installed properly, then there are different ways in which you can run this project. If you just want to run the project using simply web cam then run this command

```bash
$ python3 run.py
```

If you want to run a sample video file, the at first save the video in the `Sample_video` folder and suppose the video name `Warrior2.mp4` then for running that for that specific file, just run this commmand.

```
$ python3 run.py  --vid_name Warrior2.mp4
```

If you want to save the results the just add one more argument like this:

```
$ python3 run.py  --vid_name Warrior2.mp4 --save_as warrior_result.mp4
```
This will save the resultant video on the directory: `Video_result` automatically.

**Running the results in streamlit Local host.**

For running the app in local host using streamlit just type.

```bash
$ streamlit run app.py
```
And this will run the webapp on the localhost


----
<br>


<a name="re_train_model"></a>
## **How to train a new model or retrain the existing model**

In order to do that, the run this scipt:
```py

import torch
import torch_geometric 

from src.yoga_pos_train import YogaPoseTrain
from Models.base_gnn_model import Model 

yoga_train = YogaPoseTrain()
train_loader =  yoga_train.load_data_loader("train_data.csv")
test_loader = yoga_train.load_data_loader("test_data.csv")

# either retrain the base model using this

model = Model(3, 64, 16, 5)

# or make a new Graph Model and use that

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

yoga_train.train(model, criterion, optimizer, 30, train_loader, test_loader)
```

This will train the model and will save the weights on the `saved_models` directory. 

---

<a name="results"></a>
## **Results**

Intial results of the model is pretty good based on the fact it has relatively less training data as its just 250 images (per class). After training the model is giving a train accuracy of `0.89` and a test accuracy of `0.91`. This could be improved by improving the model and adding edge features also. This model is also using limited nodes, as I did't add the face and the palm nodes, in order to make it more realistic. 

---
<br>

## **Improvements done after release of version 1.0**

**Software improvements**

1. Improved the docker file and solved the issue of streamlit port problem
2. Set up GitHub actions. Implemented CI/CD pipeline for docker. So every merge to `main` branch will result in improvement in the docker image directly in the DockerHub.

**ML Model Improvements**

1. Minor changes in model, tweaked some hyper parameters and implemented BatchNorm. 
2. GraphAttention Conv increased it's test accuracy by 2 %/

---

<br>

<a name="future_works"></a>
## **Future Works**

Currently there are some issues both interms of the Software infrastructure and the AI model. Both of them has a great scope of developement. Some of the works that are currently taking place and can be done in future are listed below. 

**Improvements based on software developement**

1. Implementing better CI/CD pipeline that will directly deploy the docker image to DockerHub and required changes to Heroku.

2. Solving the problem of providing OS privilages like camera acess to the user while running on heroku. 

**Model based future works and potential improvements**

1. Finding a proper video based dataset providing different actions of Yoga poses. 
2. Implemeting Attention based TGN models for better and much robust performance. 
3. Implementing a mechanism, that will help the user through guiding the switch between one pose to another pose.
4. An AI based yoga assitant that will asses how much better the user is doing all the poses and what can be done. This can be potentially done by fusing NLP, XAI and Existing Approaches of Computer Vision and Graph ML.