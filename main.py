import os
from .enhancer import ImageEnhancer
from .model import VisionModel
from .extractor import InvoiceExtractor
from google.colab import drive

def main():
    drive.mount('/content/drive')
    img_path = os.getenv('INVOICE_PATH', '/content/drive/MyDrive/invoice_dataset/my_invoice_1.jpg')

    enhancer = ImageEnhancer()
    model = VisionModel()
    extractor = InvoiceExtractor(enhancer, model)

    result = extractor.extract(img_path)
    print(result)

if __name__ == '__main__':
    main()