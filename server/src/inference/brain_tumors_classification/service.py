import torch
from torchvision import transforms
import io
from PIL import Image

from .model import BrainTumorClassifier


# This block will be explained quickly through the part 1 of the deployment mechanism
class BrainTumorClassificationService:
    """
    A service for brain tumor classification.

    Handles loading the model, preprocessing input images, and making predictions.
    """

    def __init__(self, model_path: str):
        """
        Initialize the classification service.

        Args:
            model_path (str): Path to the pre-trained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def _load_model(self, model_path: str):
        """
        Load the pre-trained model from the specified path.

        Args:
            model_path (str): Path to the model file.

        Returns:
            BrainTumorClassifier: The loaded model.
        """
        model = BrainTumorClassifier(num_classes=4)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        return model

    def predict(self, image_bytes: bytes):
        """
        Predict the tumor type from the input image.

        Args:
            image_bytes (bytes): Byte representation of the image.

        Returns:
            dict: Predicted tumor class and its probability.
        """
        # Load and preprocess the image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert grayscale to RGB if necessary
        if image.mode != "RGB":
            image = image.convert("RGB")

        image = self.transform(image).unsqueeze(0).to(self.device)

        # Perform prediction
        with torch.no_grad():
            output = self.model(image)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            top_prob, predicted = torch.max(probabilities, 1)

        # Map the prediction to class names
        predicted_class = self.class_names[predicted.item()]
        probability = top_prob.item()

        return {"class": predicted_class, "probability": probability}
