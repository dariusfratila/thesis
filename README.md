# Romanian Visual Speech Recognition

## Overview
This repository contains the implementation of a **lipreading pipeline** designed for **visual speech recognition**. The project processes video frames of mouth movements to predict spoken words using advanced deep learning models.

---

## Features
- **Backend API** built with Flask for video processing and model inference.
- **YOLO-based mouth detection** for accurate region-of-interest extraction.
- **Temporal deep learning models** for lipreading using 3D CNNs and TCNs.
- **Saliency map generation** to visualize model attention.
- Modular and scalable pipeline design with a focus on maintainability.

---

## Code Structure
The repository is organized as follows:

```plaintext
LIPREADING_PIPELINE/
│
├── api/                          # Backend API for video processing and predictions
│   └── lipreading_api_server.py  # Flask server for handling video uploads and processing
│
├── config/                       # Configuration files for the project
│   └── project_config.py         # Centralized configuration for paths and settings
│
├── backbone/                     # Core deep learning model architectures
│   ├── feature_lateral_inhibition.py  # Lateral inhibition module for feature interaction
│   ├── model_loader.py                # Model initialization and weight loading
│   └── temporal_multiscale_model.py   # Temporal models for lipreading
│
├── trained_models/               # Pretrained models 
│   ├── mouth_detection_yolo_v1.pt    # YOLO model for mouth detection
│   ├── lipreading_model_v1.pt        # Lipreading model (version 1)
│   └── lipreading_model_v2_128.pt    # Lipreading model (version 2)
│
├── processing/                   # Processing modules for motion analysis and utilities
│   ├── bbox_calculations.py      # Bounding box calculations for mouth regions
│   ├── data_processing_utils.py  # Utility functions for video and frame processing
│   ├── logging_config.py         # Logging configuration
│   ├── motion_analysis.py        # Motion-based frame selection
│   └── mouth_frame_extractor.py  # Video frame extraction and mouth detection
│
└── README.md                     
