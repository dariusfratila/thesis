import torch

from backbone.temporal_multiscale_model import LipReadModel
from torchvision import transforms
from config import project_config


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class LipReadingModel:
    """
    Wrapper class for the Lip Reading Model to handle initialization,
    preprocessing, and predictions.

    Attributes:
        model (LipReadModel): The Lip Reading Model.
        index_to_word (dict): Mapping from class indices to word labels.
    """

    def __init__(self, index_to_word):
        """
        Initializes the LipReadingModel with the specified index-to-word mapping.

        Args:
            index_to_word (dict): A dictionary mapping class indices to word labels.
        """
        self.model = LipReadModel(num_classes=19)
        self.model.load_state_dict(torch.load(
            project_config.Config.MODEL_PATH, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.index_to_word = index_to_word

    @staticmethod
    def transform():
        """
        Returns a transformation pipeline to preprocess image frames.

        Returns:
            torchvision.transforms.Compose: The transformation pipeline.
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
        Predicts the top-k words based on the given input tensor of frames.

        Args:
            frames_tensor (torch.Tensor): A tensor of frames to predict from.
                - Expected shape: (batch_size, channels, depth, height, width) for 5D tensors.
                - Expected shape: (channels, depth, height, width) for 4D tensors.
            return_grad (bool): If True, returns the gradient of the input tensor for saliency maps.

        Returns:
            list of tuple: A list of top-k words with their corresponding probabilities.
            torch.Tensor (optional): Gradients of the input tensor if `return_grad=True`.
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
