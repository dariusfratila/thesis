class Config:
    """
    A configuration class containing project-wide constants and settings.

    Attributes:
        UPLOAD_FOLDER (str): Directory for uploading video files.
        MOUTH_FRAMES_FOLDER (str): Directory for storing extracted mouth frames.
        ALLOWED_EXTENSIONS (set): Permitted video file extensions for uploads.
        MAX_CONTENT_LENGTH (int): Maximum file size allowed for uploads (in bytes).
        MODEL_PATH (str): Path to the trained LipReading model.
        YOLO_MODEL_PATH (str): Path to the YOLO model for mouth detection.
        MOTION_THRESHOLD (int): Threshold for detecting motion in video frames.
        FULL_FRAMES_FOLDER (str): Directory for storing full processed frames.
    """

    UPLOAD_FOLDER = 'data/uploaded_videos'
    """Directory where uploaded videos are stored."""

    MOUTH_FRAMES_FOLDER = 'data/mouth_frames'
    """Directory where extracted mouth frames are saved."""

    ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov'}
    """Allowed file extensions for video uploads."""

    MAX_CONTENT_LENGTH = 16 * 1024 * 1024
    """Maximum size of uploaded files in bytes (16 MB)."""

    MODEL_PATH = 'trained_models/lipreading_model_v1.pt'
    """Path to the primary trained LipReading model (version 1)."""

    # Alternative model version
    # MODEL_PATH = 'models/lipreading_model_v2.pt'

    YOLO_MODEL_PATH = 'trained_models/mouth_detection_yolo_v1.pt'
    """Path to the YOLO model for mouth detection."""

    MOTION_THRESHOLD = 5000
    """Threshold for motion detection in video frames."""

    FULL_FRAMES_FOLDER = 'data/full_frames'
    """Directory where full frames are saved after processing."""
