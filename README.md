# **LipReading AI Application**

## **Overview**

This project implements a lipreading system that processes video inputs and performs lipreading predictions using a deep learning model. The repository includes data processing modules, models, and an API for performing lipreading. The repository is organized into the following main directories:

- `data/`: Contains datasets and processed video frames.
- `modules/`: Contains the implementation of models, video processing, and utility logic.
- `config/`: Stores the application configuration settings.
- `weights/`: Contains pre-trained model weights for lipreading and YOLO.

---

## **Directory Structure**

### **Config**
- **config/**
  - `app_config.py`: Contains the configuration settings for paths, thresholds, and other parameters used throughout the application.

---

### **Data**
- **data/**
  - `dataset/`: Contains the validation dataset (`val_20`).
  - `full_frames/`: Stores full frames extracted from videos for visualization.
  - `mouth_frames/`: Stores extracted mouth frames after processing video inputs.
  - `uploaded_videos/`: Stores videos uploaded for lipreading.

---

### **Models**
- **models/**
  - `128_final_model.pt`: An alternate pretrained model file.
    
---

### **Modules**

#### **Models**
- **modules/models/**
  - `lateral_inhibition.py`: Implements a neural network layer with lateral inhibition functionality.
  - `lipreading_inference.py`: Contains the inference pipeline, transforming frames and using the lipreading model for predictions.
  - `lipreading_model.py`: Defines the full deep learning model used for lipreading, including the temporal convolutional network (TCN) and lateral inhibition layers.

#### **Video Processing**
- **modules/video_processing/**
  - `frame_utils.py`: Contains utility functions for processing video frames, such as detecting and enhancing the mouth region.
  - `video_processing.py`: Main script for video processing, which detects the mouth region in each frame and selects key frames based on motion analysis.

#### **Utils**
- **modules/utils/**
  - `logger.py`: Sets up logging configuration for the application.

---

### **Weights**
- **weights/**
  - `final_lipreading_model.pt`: Final trained lipreading model.
  - `yolo_model_h.pt`: Pretrained YOLO model for mouth detection.

---

### **Main Application**
- `main_app.py`: Main script for the Flask API. It handles video uploads, processes the video for lipreading, and returns predictions and visual saliency maps.

---

## **API Usage**

This project provides a Flask API that can be used to upload videos, process them, and perform lipreading predictions. The predictions are accompanied by saliency maps that visualize the model’s attention.

### **Endpoints:**

1. **POST /demo**: 
   - **Description**: Upload a video for lipreading. The API will process the video and return the predicted words along with a saliency map GIF.
   - **Request Body**: A video file in `.mp4`, `.avi`, or `.mov` format.
   - **Response**: JSON object with the predictions and path to the generated saliency map.

---

## **Data Preprocessing**

The system includes tools for processing video files, extracting mouth regions, and creating frame datasets for further training or inference.

### **Video Processing Steps:**
1. **Frame Extraction**: Extracts frames from video inputs.
2. **Mouth Detection**: Detects the mouth region in each frame using a pre-trained YOLO model.
3. **Motion Analysis**: Selects key frames based on motion scores to focus on frames with higher mouth movement.

---

## **Model**

The lipreading model includes:
- **3D Convolutional Layers**: Used to process input video frames.
- **ResNet-18**: A pretrained network used to extract features from frames.
- **Multi-Scale Temporal Convolutional Network (MS-TCN)**: Handles the temporal dynamics of lip movements.
- **Lateral Inhibition Layer**: Implements lateral inhibition to enhance the prediction process.
