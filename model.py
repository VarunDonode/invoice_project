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
            load_in_4bit=True
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def infer(self, image):
        """
        Run inference on the given PIL image.

        Args:
            image (PIL.Image): The invoice image.

        Returns:
            str: Structured markdown output.
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": (
                        "You are a helpful vision-language model trained to extract and present structured information from invoice images.\n\n"
                        "Return only clean and well-formatted **Markdown text**.\n"
                        "Do not use tables. Use bold field titles and bullet points.\n\n"
                        "If any field is missing in the image, omit it entirely from the output. Do not guess or hallucinate.\n\n"
                        "### Fields to Extract (if available):\n"
                        "- Invoice Number\n"
                        "- Invoice Date\n"
                        "- Due Date\n"
                        "- Vendor Name\n"
                        "- Vendor Address\n"
                        "- Bill To\n"
                        "- Ship To\n"
                        "- Line Items (as bullet points)\n"
                        "- Subtotal\n"
                        "- Tax\n"
                        "- Total Amount Due\n"
                        "- Payment Terms\n\n"
                        "### Output:"
                    )}
                ]
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        generated = self.model.generate(**inputs, max_new_tokens=512)
        trimmed_ids = [gen[len(inp):] for inp, gen in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()
