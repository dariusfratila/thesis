import os
import logging
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio
from flask import send_from_directory, Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from processing.logging_config import configure_logging

from config.project_config import Config
from processing.data_processing_utils import allowed_file, create_index_to_word_dict
from processing.mouth_frame_extractor import VideoProcessor
from backbone.model_loader import LipReadingModel

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

classes_root = 'data/dataset/val_20'
index_to_word = create_index_to_word_dict(classes_root)
video_processor = VideoProcessor()
lip_reading_model = LipReadingModel(index_to_word)


@app.route('/images/<path:filename>')
def serve_image(filename):
    """
    Serve an image from the upload directory.
    """
    directory = os.path.dirname(filename)
    filename = os.path.basename(filename)
    full_path = os.path.join(app.config['UPLOAD_FOLDER'], directory)
    logging.info(f"Serving file {filename} from {full_path}")
    return send_from_directory(full_path, filename, mimetype='image/gif')


@app.route('/demo', methods=['POST'])
@cross_origin()
def upload_file():
    """
    Handle file upload, process the video, and generate predictions and saliency maps.
    """
    logging.info("Received a POST request to /demo.")
    file = request.files.get('file')

    if not file or file.filename == '':
        logging.warning("No file selected.")
        return jsonify({'message': 'No selected file'}), 400

    if not allowed_file(file.filename):
        logging.warning("Invalid file type.")
        return jsonify({'message': 'Invalid file type'}), 400

    filename = secure_filename(file.filename)  # type: ignore
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    logging.info(f"File saved at {file_path}")

    try:
        mouth_frames_folder = video_processor.process_video(file_path)
        if not mouth_frames_folder:
            logging.warning("No mouth detected in the video.")
            return jsonify({'message': 'No mouth detected in video'}), 404

        frames_tensor = video_processor.load_and_transform_frames(
            mouth_frames_folder)
        predictions, saliency_maps = video_processor.get_saliency_maps(
            frames_tensor, lip_reading_model)

        saliency_folder, gif_output_path = generate_saliency_outputs(
            filename, mouth_frames_folder, saliency_maps
        )

        result_data = {
            'message': 'File uploaded and processed successfully',
            'predictions': predictions,
            'saliency_folder': saliency_folder,
            'saliency_maps_gif': gif_output_path,
        }
        return jsonify(result_data), 200

    except Exception as e:
        logging.error(f"Error processing video: {str(e)}")
        return jsonify({'message': 'Error processing video', 'error': str(e)}), 500


def generate_saliency_outputs(filename, mouth_frames_folder, saliency_maps):
    """
    Generate saliency map outputs and save them as images and GIFs.

    Args:
        filename (str): Original file name.
        mouth_frames_folder (str): Folder containing mouth frames.
        saliency_maps (list): List of saliency maps.

    Returns:
        tuple: Path to the saliency folder and the generated GIF file.
    """
    saliency_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_saliency_maps")
    os.makedirs(saliency_folder, exist_ok=True)

    gif_images = []
    for frame_index, saliency_map in enumerate(saliency_maps):
        saliency_map = process_saliency_map(saliency_map)

        original_frame = cv2.imread(os.path.join(
            mouth_frames_folder, f"{frame_index}.jpg"))
        heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_HOT)
        superimposed_img = cv2.addWeighted(
            original_frame, 0.8, heatmap, 0.3, 0)

        output_path = save_saliency_image(
            superimposed_img, saliency_folder, frame_index)
        gif_images.append(imageio.imread(output_path))

    gif_output_path = os.path.join(saliency_folder, "saliency_maps.gif")
    logging.info(f"Generated GIF: {gif_output_path}")
    imageio.mimsave(gif_output_path, gif_images, duration=0.1, loop=0)

    return saliency_folder, gif_output_path


def process_saliency_map(saliency_map):
    """
    Process a single saliency map by applying thresholding and normalization.

    Args:
        saliency_map (numpy.ndarray): The raw saliency map.

    Returns:
        numpy.ndarray: The processed saliency map.
    """
    thresh_value = np.percentile(saliency_map, 75)
    _, saliency_map = cv2.threshold(
        saliency_map, thresh_value, 255, cv2.THRESH_TOZERO)
    saliency_map = cv2.normalize(
        saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)  # type: ignore
    return saliency_map


def save_saliency_image(image, saliency_folder, frame_index):
    """
    Save a single saliency map image.

    Args:
        image (numpy.ndarray): The image to save.
        saliency_folder (str): Path to the saliency folder.
        frame_index (int): The index of the frame.

    Returns:
        str: Path to the saved image.
    """
    output_path = os.path.join(
        saliency_folder, f"saliency_map_{frame_index}.png")
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    return output_path


if __name__ == '__main__':
    app.run(debug=True)
