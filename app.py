from distutils.log import warn
import os
import sys 
import cv2 
import base64
import warnings
import webbrowser
import streamlit as st 

warnings.filterwarnings("ignore")

path = os.getcwd()

sys.path.append(path)
sys.path.append("..")

from src.live_detection import PoseRun

class StreamlitAppDetection(object):
    def __init__(self, device = 'cpu'):
        self.device = device 
        self.poserun = PoseRun(device=self.device)

    def __find_camera_devices(self):
        cams = [] 
        cam = 0
        while True:
            if (cv2.VideoCapture(cam).grab()) is True:
                cams.append(cam)
                cam += 1
            else:
                break
        return cams 
    
    def demo(self):
        st.title("Real time Yoga Pose APP")
        st.text("Streamlit 💚 Mediapipe 💚 PyTorch Geometric")
        cam_nums = self.__find_camera_devices()
        selected_cam = -1

        PoseRun(device='cpu').run_detection_on_app(cam_num=selected_cam)
    
    def about_project(self):
        st.title("Real time Yoga Pose APP")
        st.text("Streamlit 💚 Mediapipe 💚 PyTorch Geometric")
        
        st.markdown("## Welcome to YogaPoseGNN :smile:") 
        gif_path = os.path.join(path, 'Images/tree_PtFptRsa.gif')
        st.markdown("![Alt Text](https://media.giphy.com/media/CocmEsPoERVdDj4PS5/giphy.gif)")
        st.markdown(
            """
[GIF credits](https://www.youtube.com/watch?v=k4qaVoAbeHM&ab_channel=Howcast). This is another kinda same project done and made on the top of [SignLangGNN](https://github.com/Anindyadeep/SignLangGNN). But this project is much more robust and accurate and can perform real time Yoga position classification using Graph Neural Networks. The best part of this project is that the CPU utlization. As its just using some pixel co-ordinate changes in the video. Also Graph neural networks are emerging more and more in several aspects of computer vision. So this problem is framed as a graph classification problem. I used a simple two Graph Attention layers and a softmax classifier as the network architecture. In just 20 epochs it gives an accuracy of `0.89` and `0.86` of train and test accuracy respectively. 

----

## **How to run the project**

At first clone the project using the command:
```
git clone https://github.com/Anindyadeep/YogaPosGNN.git
```

This project is using **OpenCV**, **TensorFlow**, **PyTorch**, **PyTorch Geometric**, and **mediapipe**.

If you want to run the project using simply web cam then run this command
```
python3 run.py
```

If you want to run a sample video file, the at first save the video in the `Sample_video` folder and suppose the video name `Warrior2.mp4` then for running that for that specific file, just run this commmand.

```
python3 run.py  --vid_name Warrior2.mp4
```

If you want to save the results the just add one more argument like this:

```
python3 run.py  --vid_name Warrior2.mp4 --save_as warrior_result.mp4
```
This will save the resultant video on the directory: `Video_result` automatically.

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

## **How to run the webapp**

Just type:
```bash
streamlit run app.py
```
And this will run the webapp on the localhost

### **Build and run docker image**

To create the image type:
```
docker build -t <image_name> . 
```
Here `<image_name>` is the custom name of the image.

To run the image type:
```
docker run --privileged --device=/dev/video0:/dev/video0 <image_name>
```
This will do all the work from installation of the dependencies to running the project on localhost.


## **Results**

Intial results of the model is pretty good based on the fact it has relatively less training data as its just 250 images (per class). After training the model is giving a train accuracy of `0.86` and a test accuracy of `0.90`. This could be improved by improving the model and adding edge features also. This model is also using limited nodes, as I did't add the face and the palm nodes, in order to make it more realistic. 

---

## **Future Works**
Using **Temporal Graph Neural Nets** could make more robust and accurate model for this kind of problem. But for that we need temporal data like videos instaed of images, so that we could generate `static temporal graphs` and compute on them as a dynamic graph sequence problem.
""")

    def about_me(self):
        st.markdown(
        """
        ## Hello :wave:
        #### I am **Anindyadeep**, an evergreen student trying to reinforce himself to "learn", and optimize his loss... in order to "push" a better version of itself...
        """)
        st.markdown(
            """
            ##### I am interested in **Machine Learning**, **Deep Learning**, **GraphML**, **Computer Vision**
            """)

        st.markdown("###### Please feel free to reach out via:")
        col1, col2, col3 = st.columns([0.33, 0.33, 1])
        website = col1.button('Website')
        github = col2.button('GitHub')
        linkedin = col3.button('LinkedIn')
        webbrowser.open('https://anindyadeep.github.io/portfolio/') if website else None 
        webbrowser.open('https://github.com/Anindyadeep') if github else None 
        webbrowser.open('https://www.linkedin.com/in/anindyadeep-sannigrahi-38683b1b6') if linkedin else None 


    def run(self):
        activities = ["About the project", "Demo", "About me"]
        choice = st.sidebar.selectbox("Select", activities)

        if choice == 'About the project':
            self.about_project()
        
        elif choice == 'Demo':
            self.demo()
        
        else:
            self.about_me()
        

if __name__ == '__main__':
    StreamlitAppDetection().run()
    

