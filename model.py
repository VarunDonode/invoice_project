import torch
from transformers import AutoTokenizer, AutoModelForVision2Seq, AutoProcessor
from qwen_vl_utils import process_vision_info

class VisionModel:
    """
    Handles loading and inference with the vision-language model.
    """
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=True  # We'll upgrade this to BitsAndBytesConfig later
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def infer(self, image):
        """
        Run inference on the given PIL image.

        Args:
            image (PIL.Image): The invoice image.

        Returns:
            str: Clean Markdown output with bullet points.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "You are an intelligent model trained to extract structured information from invoice images.\n\n"
                        "✅ Output should be in plain **Markdown** using only bullet points.\n"
                        "❌ Do NOT use tables or column formatting.\n"
                        "❌ Do NOT guess missing fields. Only include fields actually present in the image.\n"
                        "❌ Avoid repeating phrases.\n\n"
                        "### Required Output Format:\n"
                        "- **Invoice Number:**\n"
                        "- **Invoice Date:**\n"
                        "- **Due Date:**\n"
                        "- **Vendor Name:**\n"
                        "- **Vendor Address:**\n"
                        "- **Bill To:**\n"
                        "- **Ship To:**\n"
                        "- **Line Items:**\n"
                        "- **Subtotal:**\n"
                        "- **Tax:**\n"
                        "- **Total Amount Due:**\n"
                        "- **Payment Terms:**\n\n"
                        "If any of the above fields is not present in the invoice, skip it silently."
                    )}
                ]
            }
        ]

        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        # Reduce tokens to avoid runaway generation
        generated = self.model.generate(**inputs, max_new_t**
