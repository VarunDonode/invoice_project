import argparse
from enhancer import ImageEnhancer
from model import VisionModel
from extractor import InvoiceExtractor

def main():
    parser = argparse.ArgumentParser(description="Extract invoice fields from an image.")
    parser.add_argument('--img_path', type=str, required=True, help='Path to the invoice image')
    args = parser.parse_args()

    enhancer = ImageEnhancer()
    model = VisionModel()
    extractor = InvoiceExtractor(enhancer, model)

    result = extractor.extract(args.img_path)
    print("\nPutput:-\n")
    print(result)

if __name__ == '__main__':
    main()
