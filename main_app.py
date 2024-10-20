import logging
import os
from modules.video_processing.video_processing import VideoProcessor

import cv2
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import imageio

from flask import send_from_directory, Flask, request, jsonify
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
from config.app_config import Config
from modules.video_processing.frame_utils import allowed_file, create_index_to_word_dict


from modules.models.lipreading_inference import LipReadingModel

matplotlib.use('Agg')

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
    Serve static images (e.g., GIFs) from the specified directory.

    Args:
        filename (str): The filename of the image to serve.

    Returns:
        Flask Response: A response object serving the requested image.
    """
    directory = os.path.dirname(filename)
    filename = os.path.basename(filename)

    full_path = os.path.join(app.config['UPLOAD_FOLDER'], directory)
    logging.info(f"Serving from {full_path} the file {filename}")
    return send_from_directory(full_path, filename, mimetype='image/gif')


@app.route('/demo', methods=['POST'])
@cross_origin()
def upload_file():
    """
    Handle video uploads for lipreading demo.

    This endpoint accepts a video file, processes it to extract mouth frames, generates saliency maps
    using the lipreading model, and returns the processed results along with a generated GIF.

    Returns:
        Flask Response: A JSON response with the prediction results and paths to the saliency maps.
    """
    print("\nReceived a POST request.\n" + "-"*40)
    file = request.files.get('file')
    if not file or file.filename == '':
        logging.warning("No file selected.")
        return jsonify({'message': 'No selected file'}), 400

    if allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logging.info(f"File saved at {file_path}")

        try:
            mouth_frames_folder = video_processor.process_video(file_path)
            if mouth_frames_folder is None:
                logging.warning("No mouth detected in video.")
                return jsonify({'message': 'No mouth detected in video'}), 404

            frames_tensor = video_processor.load_and_transform_frames(
                mouth_frames_folder)
            predictions, saliency_maps = video_processor.get_saliency_maps(
                frames_tensor, lip_reading_model)

            saliency_folder = os.path.join(app.config['UPLOAD_FOLDER'], f"{os.path.splitext(filename)[0]}_saliency_maps")
            os.makedirs(saliency_folder, exist_ok=True)

            gif_images = []
            for frame_index, saliency_map in enumerate(saliency_maps):
                thresh_value = np.percentile(saliency_map, 75)
                _, saliency_map = cv2.threshold(
                    saliency_map, thresh_value, 255, cv2.THRESH_TOZERO)

                saliency_map = cv2.normalize(
                    saliency_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)

                original_frame = cv2.imread(os.path.join(
                    mouth_frames_folder, f"{frame_index}.jpg"))
                heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_HOT)
                superimposed_img = cv2.addWeighted(
                    original_frame, 0.8, heatmap, 0.3, 0)

                plt.figure()
                plt.imshow(cv2.cvtColor(superimposed_img, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                output_path = os.path.join(
                    saliency_folder, f"saliency_map_{frame_index}.png")
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
                plt.close()

                gif_images.append(imageio.imread(output_path))

            gif_output_path = os.path.join(
                saliency_folder, "saliency_maps.gif")
            logging.info(f"Generated GIF: {gif_output_path}")
            imageio.mimsave(gif_output_path, gif_images, duration=0.1, loop=0)

            result_data = {
                'message': 'File uploaded and processed successfully',
                'predictions': predictions,
                'saliency_folder': saliency_folder,
                'saliency_maps_gif': gif_output_path,
            }

            return jsonify(result_data), 200
        except Exception as e:
            logging.error(f"Error processing video: {str(e)}")
            print(f"Error processing video: {e}\n" + "-"*40)
            return jsonify({'message': 'Error processing video', 'error': str(e)}), 500
    else:
        logging.warning("Invalid file type.")
        return jsonify({'message': 'Invalid file type'}), 400


if __name__ == '__main__':
    app.run(debug=True)
