import os
import re
import cv2
import torch
import logging
from ultralytics import YOLO
from config import project_config
from processing.motion_analysis import analyze_motion, select_top_frames
from processing.bbox_calculations import calculate_mouth_bbox
from processing.logging_config import configure_logging
from processing.data_processing_utils import enhance_mouth_region
from backbone.model_loader import LipReadingModel
configure_logging()


class VideoProcessor:
    """
    Processes videos to extract and transform frames containing mouth regions
    for lip-reading models.
    """

    def __init__(self):
        """
        Initializes the VideoProcessor with the YOLO model for detecting mouth regions.
        """
        self.model = YOLO(project_config.Config.YOLO_MODEL_PATH)
        logging.info(
            "Initialized VideoProcessor with YOLO model loaded from %s", project_config.Config.YOLO_MODEL_PATH
        )

    def process_video(self, video_path):
        """
        Processes a video to extract mouth frames.

        Args:
            video_path (str): Path to the input video file.

        Returns:
            str: Path to the directory containing extracted mouth frames.
        """
        cap = self._open_video(video_path)
        frames, motion_scores = analyze_motion(cap)
        selected_frames = select_top_frames(frames, motion_scores)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        mouth_extract_folder = os.path.join(
            project_config.Config.MOUTH_FRAMES_FOLDER, video_name)
        full_frames_folder = os.path.join(
            project_config.Config.FULL_FRAMES_FOLDER, video_name)
        os.makedirs(mouth_extract_folder, exist_ok=True)
        os.makedirs(full_frames_folder, exist_ok=True)

        self._extract_mouth_frames(
            selected_frames, full_frames_folder, mouth_extract_folder)
        return mouth_extract_folder if len(selected_frames) == 29 else None

    def _open_video(self, video_path):
        """
        Opens a video file for processing.

        Args:
            video_path (str): Path to the video file.

        Returns:
            cv2.VideoCapture: Video capture object.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Failed to open video file: %s", video_path)
            raise Exception("Error opening video file")
        logging.info("Processing video: %s", video_path)
        return cap

    def _extract_mouth_frames(self, frames, full_frames_folder, mouth_extract_folder):
        """
        Extracts mouth regions from selected frames and saves them.

        Args:
            frames (list): List of selected video frames.
            full_frames_folder (str): Directory to save full frames.
            mouth_extract_folder (str): Directory to save mouth regions.
        """
        for i, frame in enumerate(frames):
            bbox = self.extract_mouth_bbox(frame)
            if bbox:
                x1, y1, x2, y2 = bbox
                mouth_region = frame[y1:y2, x1:x2]
                mouth_region_resized = cv2.resize(
                    mouth_region, (64, 64), interpolation=cv2.INTER_CUBIC)
                mouth_region_resized = enhance_mouth_region(
                    mouth_region_resized)
                cv2.imwrite(os.path.join(mouth_extract_folder,
                            f"{i}.jpg"), mouth_region_resized)
                logging.info("Mouth detected and extracted in frame %d", i)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            else:
                logging.info("No mouth detected in frame %d", i)
            cv2.imwrite(os.path.join(full_frames_folder, f"{i}.jpg"), frame)

    def extract_mouth_bbox(self, frame):
        """
        Extracts the bounding box of the mouth region.

        Args:
            frame (numpy.ndarray): A single video frame.

        Returns:
            list: Coordinates [x1, y1, x2, y2].
        """
        results = self.model(frame)
        if not results or not results[0].keypoints or not results[0].keypoints.conf:
            return None

        keypoints_list = results[0].keypoints.xy.cpu().numpy()
        confidences_list = results[0].keypoints.conf.cpu().numpy()

        return calculate_mouth_bbox(frame.shape[1], keypoints_list, confidences_list)

    def load_and_transform_frames(self, frames_folder):
        """
        Load and apply transformations to extracted mouth frames.

        Args:
            frames_folder (str): Path to the folder containing mouth frames.

        Returns:
            torch.Tensor: A tensor of transformed frames ready for prediction.
        """
        transform = LipReadingModel.transform()
        frame_paths = sorted(
            [os.path.join(frames_folder, f) for f in os.listdir(
                frames_folder) if f.endswith('.jpg')],
            key=lambda x: int(re.search(r'\d+', x).group())
        )

        transformed_frames = []
        for frame_path in frame_paths:
            image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image_tensor = transform(image)
            transformed_frames.append(image_tensor.unsqueeze(0))

        return torch.cat(transformed_frames, dim=0).unsqueeze(0)

    def get_saliency_maps(self, frames_tensor, lip_reading_model):
        """
        Generates saliency maps for the given frames using the lip-reading model.

        Args:
            frames_tensor (torch.Tensor): Tensor of transformed frames.
            lip_reading_model (LipReadingModel): The lip-reading model.

        Returns:
            tuple: Predictions and saliency maps.
        """
        predictions, gradients = lip_reading_model.predict(
            frames_tensor, return_grad=True)
        gradients = gradients.squeeze()
        saliency_maps = gradients.cpu().detach().numpy()
        return predictions, saliency_maps
