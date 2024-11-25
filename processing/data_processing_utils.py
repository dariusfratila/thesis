import os
import re
import cv2
from config import project_config


def allowed_file(filename):
    """
    Check if a given file has an allowed extension based on the Config.

    Args:
        filename (str): The name of the file to check.

    Returns:
        bool: True if the file has an allowed extension, False otherwise.
    """
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in project_config.Config.ALLOWED_EXTENSIONS


def enhance_mouth_region(mouth_region):
    """
    Enhance the mouth region by converting it to grayscale.

    Args:
        mouth_region (numpy.ndarray): The mouth region image as a NumPy array.

    Returns:
        numpy.ndarray: The enhanced mouth region in grayscale.
    """
    if len(mouth_region.shape) == 3 and mouth_region.shape[2] == 3:
        mouth_region = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2GRAY)
    return mouth_region


def create_index_to_word_dict(root_dir):
    """
    Create a dictionary mapping class indices to class names based on directory contents.

    Args:
        root_dir (str): The path to the directory containing class folders.

    Returns:
        dict: A mapping of class indices (int) to class names (str).
    """
    classes = sorted(
        os_item for os_item in os.listdir(root_dir) if not os_item.startswith('.')
    )
    return {index: class_name for index, class_name in enumerate(classes)}
