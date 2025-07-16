import torch
from transformers import (
    AutoTokenizer,
    AutoModelForVision2Seq,
    AutoProcessor,
    BitsAndBytesConfig,
    GenerationConfig,
)
from qwen_vl_utils import process_vision_info

class VisionModel:
    """
    Handles loading and inference with the vision-language model.
    """
    def __init__(self):
        self.model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

        # Quantization config (fixes 4-bit warning)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            device_map="auto",
            torch_dtype=torch.float16,
            quantization_config=bnb_config,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Optional: Set temperature here cleanly
        self.generation_config = GenerationConfig(
            max_new_tokens=400,
            temperature=0.7,
            do_sample=False
        )

    def infer(self, image):
        """
        Run inference on the given PIL image using a simplified prompt.

        Args:
            image (PIL.Image): The invoice image.

        Returns:
            str: Extracted information, each field on a new line.
        """
        prompt = (
            "You are a smart assistant who specializes in understanding invoices. Below is an invoice image.\n"
            "<image>\n"
            "Extract the following fields from this invoice:\n"
            "1. Invoice Number\n"
            "2. Invoice Date\n"
            "3. Due Date\n"
            "4. Vendor Name\n"
            "5. Vendor Contact Information\n"
            "6. Customer Name\n"
            "7. Customer Address\n"
            "8. Payment Instructions\n"
            "9. Total Amount Due\n"
            "If any field is missing, say 'Not found'. Return each field on a new line."
        )

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
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

        generated = self.model.generate(
            **inputs,
            generation_config=self.generation_config
        )

        trimmed_ids = [gen[len(inp):] for inp, gen in zip(inputs.input_ids, generated)]
        return self.processor.batch_decode(trimmed_ids, skip_special_tokens=True)[0].strip()
