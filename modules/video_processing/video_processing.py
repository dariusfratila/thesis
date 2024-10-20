import os
import cv2
import numpy as np
import torch
import re
from ultralytics import YOLO
from config.app_config import Config
from modules.video_processing.frame_utils import enhance_mouth_region
from modules.models.lipreading_inference import LipReadingModel
from modules.utils.logger import setup_logging
import logging

setup_logging()


class VideoProcessor:
    """
    Video Processor for LipReading Application.

    This class handles the video processing pipeline for lipreading, including mouth region detection,
    motion analysis, and frame selection. It uses a YOLO model for keypoint detection and
    optical flow for motion-based frame selection.

    Attributes:
        model (YOLO): Pretrained YOLO model used for detecting keypoints in video frames.
    """

    def __init__(self):
        """
        Initialize the VideoProcessor with the YOLO model for mouth keypoint detection.
        """
        self.model = YOLO(Config.YOLO_MODEL_PATH)
        logging.info(
            "Initialized VideoProcessor with model loaded from %s", Config.YOLO_MODEL_PATH)

    def process_video(self, video_path: str):
        """
        Process the video to extract mouth frames based on detected keypoints and motion.

        Args:
            video_path (str): Path to the input video file.

        Returns:
            str: Path to the folder containing extracted mouth frames if successful, or None if unsuccessful.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error("Failed to open video file: %s", video_path)
            raise Exception("Error opening video file")

        logging.info("Processing video: %s", video_path)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        mouth_extract_folder = os.path.join(
            Config.MOUTH_FRAMES_FOLDER, video_name)
        os.makedirs(mouth_extract_folder, exist_ok=True)

        motion_scores = []
        frames = []
        prev_gray = None

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is None:
                prev_gray = gray
                continue

            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_score = np.sum(magnitude)
            motion_scores.append(motion_score)

            frames.append(frame)
            prev_gray = gray

        cap.release()

        top_indices = np.argsort(motion_scores)[-29:]
        selected_frames = [frames[i] for i in sorted(top_indices)]
        logging.info(
            "Selected top %d frames based on motion analysis", len(selected_frames))

        full_frames_folder = os.path.join(
            Config.FULL_FRAMES_FOLDER, video_name)
        os.makedirs(full_frames_folder, exist_ok=True)

        for i, frame in enumerate(selected_frames):
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

        return mouth_extract_folder if len(selected_frames) == 29 else None

    def extract_mouth_bbox(self, frame):
        """
        Extract the bounding box around the mouth from the given frame using YOLO.

        Args:
            frame (numpy.ndarray): Input frame from the video.

        Returns:
            list: Coordinates of the mouth bounding box [min_x, min_y, max_x, max_y].
        """
        results = self.model(frame)
        if len(results) == 0 or results[0].keypoints is None or results[0].keypoints.conf is None:
            return None

        first_result = results[0]
        keypoints = first_result.keypoints

        keypoints_list = keypoints.xy.cpu().numpy()
        confidences_list = keypoints.conf.cpu().numpy()

        frame_width = frame.shape[1]
        return self.calculate_mouth_bbox(frame_width, keypoints_list, confidences_list)

    def calculate_mouth_bbox(self, frame_width, keypoints_list, confidences_list, conf_threshold=0.8, margin=30, top_margin_reduction=60):
        """
        Calculate the bounding box around the mouth region based on detected keypoints.

        Args:
            frame_width (int): The width of the frame.
            keypoints_list (numpy.ndarray): Array of keypoint coordinates.
            confidences_list (numpy.ndarray): Array of confidence values for the keypoints.
            conf_threshold (float, optional): Confidence threshold for valid keypoints. Default is 0.8.
            margin (int, optional): Margin to add around the detected mouth region. Default is 30.
            top_margin_reduction (int, optional): Reduction in the top margin for the bounding box. Default is 60.

        Returns:
            list: Coordinates of the mouth bounding box [min_x, min_y, max_x, max_y], or None if not detected.
        """
        if keypoints_list.size == 0 or confidences_list.size == 0:
            return None

        valid_indices = confidences_list > conf_threshold
        valid_keypoints = keypoints_list[valid_indices]

        if valid_keypoints.size == 0:
            return None

        min_x = int(valid_keypoints[:, 0].min()) - margin
        min_y = int(valid_keypoints[:, 1].min()) - \
            margin + top_margin_reduction
        max_x = int(valid_keypoints[:, 0].max()) + margin
        max_y = int(valid_keypoints[:, 1].max()) + margin

        width = max_x - min_x
        height = max_y - min_y
        if width > height:
            diff = width - height
            min_y -= diff // 2
            max_y += diff // 2
        elif height > width:
            diff = height - width
            min_x -= diff // 2
            max_x += diff // 2

        slight_right_shift = 28
        if min_x - slight_right_shift >= 0:
            min_x -= slight_right_shift
            max_x -= slight_right_shift
        else:
            min_x = 0
            max_x = width

        min_y = max(min_y, 0)
        return [min_x, min_y, max_x, max_y]

    def load_and_transform_frames(self, frames_folder):
        """
        Load and transform the extracted mouth frames for lipreading prediction.

        Args:
            frames_folder (str): Path to the folder containing extracted mouth frames.

        Returns:
            torch.Tensor: Tensor containing the transformed frames, ready for model inference.
        """
        transform = LipReadingModel.transform()
        frames = sorted([os.path.join(frames_folder, f) for f in os.listdir(frames_folder) if f.endswith('.jpg')],
                        key=lambda x: int(re.search(r'\d+', x).group()))

        transformed_frames = []
        for frame_path in frames:
            image = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image_tensor = transform(image)
            transformed_frames.append(image_tensor.unsqueeze(0))
        frames_tensor = torch.cat(transformed_frames, dim=0).unsqueeze(0)
        return frames_tensor

    def get_saliency_maps(self, frames_tensor, lip_reading_model):
        """
        Generate saliency maps for the input frames using the lipreading model.

        Args:
            frames_tensor (torch.Tensor): Input tensor of mouth frames.
            lip_reading_model (LipReadingModel): The lipreading model used for inference.

        Returns:
            tuple: Predicted words and corresponding saliency maps for each frame.
        """
        predictions, gradients = lip_reading_model.predict(
            frames_tensor, return_grad=True)
        gradients = gradients.squeeze()

        saliency_maps = gradients.cpu().detach().numpy()
        return predictions, saliency_maps
