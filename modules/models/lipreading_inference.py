from modules.models.lipreading_model import LipReadModel
from modules.video_processing.frame_utils import create_index_to_word_dict
import torch
from torchvision import transforms
from config.app_config import Config

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LipReadingModel:
    """
    Wrapper class for the LipReadingModel, handling model loading, inference, and transformations.

    This class provides methods to load the lipreading model, preprocess video frames, and
    perform predictions on them. It also includes the option to return gradients for saliency map generation.

    Args:
        index_to_word (dict): Dictionary mapping class indices to corresponding word labels.

    Attributes:
        model (torch.nn.Module): The lipreading model.
        index_to_word (dict): A dictionary that maps class indices to words.
    """

    def __init__(self, index_to_word):
        """
        Initializes the LipReadingModel by loading the pretrained model, setting it to evaluation mode,
        and preparing the index-to-word dictionary for prediction output.

        Args:
            index_to_word (dict): Dictionary mapping class indices to corresponding word labels.
        """
        self.model = LipReadModel(num_classes=19)
        self.model.load_state_dict(torch.load(
            Config.MODEL_PATH, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.index_to_word = index_to_word

    @staticmethod
    def transform():
        """
        Returns the transformation pipeline to preprocess video frames for the lipreading model.

        This method defines the transformations required to resize, convert to grayscale,
        and normalize the video frames before feeding them into the model.

        Returns:
            torchvision.transforms.Compose: The composition of transformations applied to the video frames.
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485], std=[0.229])
        ])

    def predict(self, frames_tensor, return_grad=False):
        """
        Performs lipreading prediction on the input tensor of video frames.

        This method feeds the preprocessed video frames into the lipreading model, computes the probabilities
        for the top 5 predicted words, and optionally returns the gradients for saliency map visualization.

        Args:
            frames_tensor (torch.Tensor): The input video frames tensor, either 4D or 5D (batch x channels x depth x height x width).
            return_grad (bool): Whether to return gradients for saliency map generation. Default is False.

        Returns:
            list: A list of tuples containing the top 5 predicted words and their corresponding probabilities.
            torch.Tensor (optional): If `return_grad=True`, returns the gradients of the input frames.
        """
        if frames_tensor.dim() == 5:
            frames_tensor = frames_tensor.permute(0, 2, 1, 3, 4)
        if frames_tensor.dim() == 4:
            frames_tensor = frames_tensor.unsqueeze(0)

        frames_tensor.requires_grad_(return_grad)

        with torch.set_grad_enabled(return_grad):
            outputs = self.model(frames_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            if return_grad:
                target_class = outputs.argmax(dim=1)
                outputs[:, target_class].sum().backward()

        topk_probs, topk_indices = torch.topk(probabilities, 5)
        topk_words = [self.index_to_word[int(index)]
                      for index in topk_indices[0].cpu().numpy()]
        topk_probs = [float(prob) for prob in topk_probs[0].tolist()]

        if return_grad:
            grad_output = frames_tensor.grad.detach()
            return list(zip(topk_words, topk_probs)), grad_output
        else:
            return list(zip(topk_words, topk_probs))

