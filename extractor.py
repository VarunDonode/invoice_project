from PIL import Image
from model import VisionModel

class InvoiceExtractor:
    """
    Extracts structured fields from an invoice image.
    
    Args:
        model (VisionModel): The VLM model handler.
    """
    def __init__(self, model):
        self.model = model

    def extract(self, img_path):
        """
        Extract fields from the image at the given path.

        Args:
            img_path (str): Path to the invoice image.

        Returns:
            str: Extracted structured text.
        """
        pil_img = Image.open(img_path).convert("RGB")
        return self.model.infer(pil_img)
