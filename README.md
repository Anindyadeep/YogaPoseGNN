# **YogaPosGNN**

![Alt Text](Images/tree_PtFptRsa.gif)

[GIF credits](https://www.youtube.com/watch?v=wdln9qWYloU&ab_channel=Howcast). This is another kinda same project done and made on the top of [SignLangGNN](https://github.com/Anindyadeep/SignLangGNN). But this project is much more robust and accurate and can perform real time Yoga position classification using Graph Neural Networks. The best part of this project is that the CPU utlization. As its just using some pixel co-ordinate changes in the video. Also Graph neural networks are emerging more and more in several aspects of computer vision. So this problem is framed as a graph classification problem. I used a simple two Graph Attention layers and a softmax classifier as the network architecture. In just 20 epochs it gives an accuracy of `0.89` and `0.86` of train and test accuracy respectively. 

----

## **How to run the project**

At first clone the project using the command:
```
git clone https://github.com/Anindyadeep/YogaPosGNN.git
```

This project is using **OpenCV**, **TensorFlow**, **PyTorch**, **PyTorch Geometric**, and **mediapipe**. So assuming correct versions are installed, now we can train our model using this command!


```
python3 main.py
```

And we want to run the model in some video or real time, just run this command:

```
python3 run.py
```


## **Future Works**
Using **Temporal Graph Neural Nets** could make more robust and accurate model for this kind of problem. But for that we need temporal data like videos instaed of images, so that we could generate `static temporal graphs` and compute on them as a dynamic graph sequence problem.