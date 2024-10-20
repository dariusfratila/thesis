import os
import re
from config.app_config import Config
import cv2


def allowed_file(filename: str) -> bool:
    """
    Check if the file has an allowed extension.

    This function checks whether the file extension is one of the allowed types,
    as specified in the application's configuration.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def enhance_mouth_region(mouth_region):
    """
    Enhance the mouth region in a video frame by converting it to grayscale.

    This function processes a given image of the mouth region by converting
    it to grayscale if it is in color, which may improve the performance of
    lipreading models by reducing noise from color channels.

    Args:
        mouth_region (numpy.ndarray): The mouth region image to be enhanced.

    Returns:
        numpy.ndarray: The enhanced mouth region image in grayscale.
    """
    if len(mouth_region.shape) == 3 and mouth_region.shape[2] == 3:
        mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    return mouth_region


def create_index_to_word_dict(root_dir):
    """
    Create a dictionary mapping class indices to word labels.

    This function scans a directory for subdirectories or files representing different classes (e.g., words),
    and creates a mapping from numerical indices to class names. This is useful for converting the output
    of classification models into human-readable labels.

    Args:
        root_dir (str): The root directory containing class folders or files.

    Returns:
        dict: A dictionary where the keys are indices (int) and the values are class names (str).
    """
    classes = sorted(os_item for os_item in os.listdir(
        root_dir) if not os_item.startswith('.'))
    return {index: class_name for index, class_name in enumerate(classes)}
