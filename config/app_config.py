import os


class Config:
    """
    Configuration class for the LipReading application.

    This class contains configuration parameters used throughout the application,
    such as folder paths for storing uploaded videos, frames, and model checkpoints,
    as well as file format restrictions and model-related parameters.
    """

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    UPLOAD_FOLDER = os.path.join(BASE_DIR, 'data', 'uploaded_videos')
    MOUTH_FRAMES_FOLDER = os.path.join(BASE_DIR, 'data', 'mouth_frames')
    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'final_lipreading_model.pt')

    YOLO_MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'yolo_model_h.pt')
    MOTION_THRESHOLD = 5000
    FULL_FRAMES_FOLDER = os.path.join(BASE_DIR, 'data', 'full_frames')
