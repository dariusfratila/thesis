def calculate_mouth_bbox(frame_width, keypoints_list, confidences_list, conf_threshold=0.8, margin=30, top_margin_reduction=60):
    """
    Calculates the bounding box of the mouth region based on detected keypoints.

    Args:
        frame_width (int): Width of the video frame.
        keypoints_list (numpy.ndarray): List of keypoints.
        confidences_list (numpy.ndarray): List of confidence scores for the keypoints.
        conf_threshold (float): Minimum confidence threshold for keypoints.
        margin (int): Margin to add around the bounding box.
        top_margin_reduction (int): Reduction to apply to the top margin.

    Returns:
        list: Bounding box coordinates [min_x, min_y, max_x, max_y] if valid keypoints are found.
        None: If no valid keypoints are detected.
    """
    valid_indices = confidences_list > conf_threshold
    valid_keypoints = keypoints_list[valid_indices]

    if valid_keypoints.size == 0:
        return None

    min_x = max(int(valid_keypoints[:, 0].min()) - margin, 0)
    min_y = max(int(valid_keypoints[:, 1].min()) -
                margin + top_margin_reduction, 0)
    max_x = int(valid_keypoints[:, 0].max()) + margin
    max_y = int(valid_keypoints[:, 1].max()) + margin

    width, height = max_x - min_x, max_y - min_y
    if width > height:
        diff = width - height
        min_y = max(min_y - diff // 2, 0)
        max_y += diff // 2
    elif height > width:
        diff = height - width
        min_x = max(min_x - diff // 2, 0)
        max_x += diff // 2

    return [min_x, min_y, max_x, max_y]
