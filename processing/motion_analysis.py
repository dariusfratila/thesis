import cv2
import numpy as np


def analyze_motion(cap):
    """
    Analyzes motion in the video frames.

    Args:
        cap (cv2.VideoCapture): Video capture object.

    Returns:
        list: List of frames from the video.
        list: Corresponding motion scores for the frames.
    """
    motion_scores = []
    frames = []
    prev_gray = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0 # type: ignore
            )
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            motion_scores.append(np.sum(magnitude))

        frames.append(frame)
        prev_gray = gray

    cap.release()
    return frames, motion_scores


def select_top_frames(frames, motion_scores, top_n=29):
    """
    Selects the top frames based on motion scores.

    Args:
        frames (list): List of video frames.
        motion_scores (list): List of motion scores corresponding to the frames.
        top_n (int): Number of top frames to select.

    Returns:
        list: Top frames sorted by motion scores.
    """
    top_indices = np.argsort(motion_scores)[-top_n:]
    return [frames[i] for i in sorted(top_indices)]
