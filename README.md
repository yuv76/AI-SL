# AI-SL - Sign Language Recognition with CNN

This project aims to bridge communication gaps using machine learning and computer vision. We developed a Sign Language Recognition System capable of interpreting hand gestures representing the English alphabet from images and video streams. The system integrates a custom Convolutional Neural Network (CNN) built with PyTorch to classify sign language symbols in real-time.

sign-language-recognition/
├── data/                     # Sign language image dataset
├── model/
│   ├── cnn.py                # Custom CNN layer implementations
│   └── train.py              # Training script
├── video_inference.py        # Live video recognition
├── chat/
│   ├── client.py             # Chat app with integrated recognition
│   └── server.py             # Server-side chat handling
├── utils/
│   └── preprocess.py         # Image preprocessing utilities
├── README.md

## Features
 - Custom CNN Model - Implemented CNN layers using PyTorch.
 - Image-Based Recognition: Trained on a dataset of English alphabet signs.
 - Video Stream Integration: Extracts sign gestures from webcam video and converts them to text.
 - Real-Time Chat App: Integrated the model with a Python-based chat application for real-time sign-to-text messaging.

## Technologies Used

## Dataset

## Run

## Results
