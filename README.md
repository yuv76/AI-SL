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
for the model - 

## Results
