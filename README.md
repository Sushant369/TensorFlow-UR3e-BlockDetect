# TensorFlow-UR3e-BlockDetect

https://github.com/Sushant369/TensorFlow-UR3e-BlockDetect/assets/72655705/9a011a09-69bb-49aa-a6ad-a0bde677afeb



[Watch the Video](https://drive.google.com/uc?export=view&id=1a2a8maQDMclrGyuuuUNYEZ1_58RSVY4O)


## Project Overview
This repository contains the code and resources for a machine learning project aimed at detecting wooden colored blocks using TensorFlow and deploying these capabilities on the UR3e robot. The project utilizes TensorFlow's object detection API to train a model capable of identifying and localizing wooden colored blocks in real-time, allowing the UR3e robot to interact with these objects based on visual input.

### Technology Used
This project extensively utilizes TensorFlow's Object Detection API, a powerful framework designed to ease the development and training of machine learning models for object detection tasks. The API offers a collection of pre-trained models along with utilities for training, evaluating, and deploying object detection models.

### Machine Learning Models

I employed transfer learning techniques to fine-tune pre-existing models on a specific dataset of wooden colored blocks. This approach significantly reduced training time and computational resources required, while maintaining high levels of accuracy.

- MobileNet SSD: Chosen for its balance between speed and accuracy, making it ideal for real-time object detection applications. Its lightweight architecture is particularly beneficial for scenarios with limited computational resources.
- EfficientDet D0: Selected for its efficiency and scalability. It delivers state-of-the-art object detection performance, where precision is paramount, without substantially increasing computational cost.

### Hardware Acceleration
The training phase of this project was accelerated using an NVIDIA RTX 3060 GPU with 8GB of VRAM. This powerful hardware enabled faster model iterations and experiments, allowing for the rapid development and refinement of the object detection system.

By utilizing this GPU, I was able to leverage its computational capabilities to drastically reduce the time required for training and inference processes, making the development cycle more efficient and enabling real-time object detection in the application.


## Features
- Real-time Object Detection: Utilizes TensorFlow to identify and localize wooden colored blocks.
- Robot Integration: Codebase includes integration with the UR3e robot for real-world application.
- Custom Dataset: Includes a dataset of wooden colored blocks for training and testing.
- Modular Code: Easy to modify for different objects or additional functionalities.

## Challenges Faced
During the development of the TensorFlow-UR3e-BlockDetect project, I encountered a significant challenge related to color detection accuracy. Specifically, the system initially misidentified red blocks as blue and vice versa. This issue stemmed from the manner in which the dataset was collected and processed.

### Issue with Color Representation
The dataset used for training the object detection model was captured and pre-processed using OpenCV. OpenCV, by default, handles images in BGR (Blue, Green, Red) format. However, TensorFlow, and many other machine learning libraries, expect image data in RGB (Red, Green, Blue) format. This discrepancy in color channel ordering led to incorrect color identification during the detection phase.

### Resolving the Color Detection Accuracy
To address this issue, I implemented a conversion process to accurately translate the image data from BGR to RGB format before feeding it into the TensorFlow model. This adjustment ensured that the colors were correctly interpreted by the model, significantly improving the accuracy of object detection and allowing the UR3e robotic arm to correctly identify and interact with the colored blocks.

This challenge underscored the importance of understanding the data preprocessing requirements of machine learning models and the need for meticulous dataset preparation. It also highlighted the crucial role of interoperability between different software tools and libraries used in AI and robotics projects.
