# AI-SL - Sign Language Recognition with CNN

This project aims to bridge communication gaps using machine learning and computer vision. We developed a Sign Language Recognition System capable of interpreting hand gestures representing the English alphabet from images and video streams. The system integrates a custom Convolutional Neural Network (CNN) built with PyTorch to classify sign language symbols in real-time.

## Features
 - Custom CNN Model - Implemented CNN layers using PyTorch.
 - Image-Based Recognition: Trained on a dataset of English alphabet signs.
 - Video Stream Integration: Extracts sign gestures from webcam video and converts them to text.
 - Real-Time Chat App: Integrated the model with a Python-based chat application for real-time sign-to-text messaging.

## Key Branches
 - main: contains CNN code - the CNN layers implementation, the training code, the training data and the final models (seperated into general model for most layers and sub model for a group of similar ones).
 - asl_chat_server: the chat app server (c++).
 - client_ASL_chat: the chat app client (python with flet).

## Technologies Used
 - Python
 - PyTorch
 - OpenCV
 - NumPy
 - flet, c++ (For chat)

## Dataset
We found a dataset of american sign language letters designed for CNN recognition - [The Research](https://www.sciencedirect.com/science/article/pii/S2666990021000471#fig0001). 
As some of the images didnt exactly worked in real time, we had to create our own. 
all shown in /mydata.

## Run
for training the model - TrainingModelPytorchCustomLayers.py. has to be run on a machine with GPU.
for the model - ‏‏openCVCamPredict_pytorch.py.
### server & client
for client - [import](#Library) the library in library-ASL_detect. handles using the CNN model.
server - server.sln project. [server branch](https://github.com/yuv76/AI-SL/tree/asl_chat_server).
client - app.py. [client branch](https://github.com/yuv76/AI-SL/tree/client_ASL_chat).

## Library
Located in library-ASL_detect. an easy way to use the model and predict.
installation:
pip install git+https://github.com/yuv76/AI-SL.git@main#subdirectory=library-ASL_detect

use:
import ASL_detect

## Demonstrations
(both are links to youtube videos)
### Model
  [![ezgif-101319c1157467](https://github.com/user-attachments/assets/886b500e-9663-4f31-aca3-e86bc4f257af)](https://youtu.be/1_T0wYiwtE0)

### Chat
  [![chat](https://github.com/user-attachments/assets/d17282cd-6e36-4306-9171-6bbffd4abd09)](https://youtu.be/-lfapyNCSpA)
