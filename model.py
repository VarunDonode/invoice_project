import torch
from transformers import AutoProcessor, AutoTokenizer, AutoModelForVision2Seq


class VisionModel:
    """
    Wrapper for Qwen2.5-VL inference.

    Args:
        model_id (str): Hugging Face model repo ID.
        device (str | None): computation device (e.g., 'cuda' or 'cpu').
        load_in_4bit (bool): whether to load weights in 4-bit.
    """
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str | None = None,
        load_in_4bit: bool = True,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_4bit=load_in_4bit,
        )

    def infer(self, messages: list[dict], max_new_tokens: int = 512) -> str:
        """
        Run a chat-style vision-to-text generation.

        Args:
            messages (list[dict]): each with 'role' and 'content', where content is a list of dicts
                representing text or images.
            max_new_tokens (int): maximum tokens to generate.

        Returns:
            str: decoded text output.
        """
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = __import__(
            'qwen_vl_utils', fromlist=['process_vision_info']
        ).process_vision_info(messages)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        streamer = __import__('transformers', fromlist=['TextStreamer']).TextStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=True
        )
        gen = self.model.generate(
            **inputs, max_new_tokens=max_new_tokens, streamer=streamer
        )

        trimmed = [out[len(inputs.input_ids[0]) :] for out in gen]
        return self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]