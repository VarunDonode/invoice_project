import cv2
import numpy as np
from PIL import Image, ImageEnhance


class ImageEnhancer:
    """
    Enhance invoice images: denoise, sharpen, resize, and adjust contrast.

    Args:
        target_size (tuple[int, int]): output image dimensions (width, height).
        denoise_h (int): filter strength for luminance component.
        denoise_h_color (int): filter strength for color components.
        sharpen_kernel (list[list[int]]): convolution kernel for sharpening.
        contrast_factor (float): factor for contrast enhancement.
    """
    def __init__(
        self,
        target_size: tuple[int, int] = (1280, 960),
        denoise_h: int = 10,
        denoise_h_color: int = 10,
        sharpen_kernel: list[list[int]] = [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
        contrast_factor: float = 1.8,
    ):
        self.target_size = target_size
        self.denoise_h = denoise_h
        self.denoise_h_color = denoise_h_color
        self.sharpen_kernel = np.array(sharpen_kernel)
        self.contrast_factor = contrast_factor

    def enhance(self, pil_img: Image.Image) -> Image.Image:
        """
        Apply enhancement pipeline to a PIL image.

        Args:
            pil_img (Image.Image): input image in RGB mode.

        Returns:
            Image.Image: enhanced image.
        """
        # to BGR numpy array
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        img = cv2.resize(img, self.target_size)
        img = cv2.fastNlMeansDenoisingColored(
            img, None, self.denoise_h, self.denoise_h_color, 7, 21
        )
        img = cv2.filter2D(img, -1, self.sharpen_kernel)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_out = Image.fromarray(rgb)
        enhancer = ImageEnhance.Contrast(pil_out)
        return enhancer.enhance(self.contrast_factor)