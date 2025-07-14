from PIL import Image
from enhancer import ImageEnhancer
from model import VisionModel


class InvoiceExtractor:
    """
    High-level interface to load, enhance, and extract invoice fields.

    Args:
        enhancer (ImageEnhancer): instance of image enhancer.
        model (VisionModel): instance of vision-language model.
    """
    def __init__(self, enhancer: ImageEnhancer, model: VisionModel):
        self.enhancer = enhancer
        self.model = model

    def extract(self, img_path: str) -> str:
        """
        Run full pipeline: load, enhance, and infer.

        Args:
            img_path (str): path to invoice image.

        Returns:
            str: model output with extracted fields.
        """
        img = Image.open(img_path).convert("RGB")
        enhanced = self.enhancer.enhance(img)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": enhanced},
                    {"type": "text", "text": (
                        "Below is an invoice image.\n"
                        "Extract the following fields:\n"
                        "1. Invoice Date\n"
                        "2. Company Name\n"
                        "3. Total Amount\n"
                        "4. Payment Terms\n"
                        "5. Shipper Address\n"
                        "6. Receiver Address\n"
                        "If any field is missing, say 'Not found'."
                    )}
                ],
            }
        ]
        return self.model.infer(messages)
