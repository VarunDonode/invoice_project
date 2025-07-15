import argparse
from enhancer import ImageEnhancer
from model import VisionModel
from extractor import InvoiceExtractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    args = parser.parse_args()

    extractor = InvoiceExtractor(
        enhancer=ImageEnhancer(),
        model=VisionModel()
    )

    result = extractor.extract(args.img_path)
    print(result)

if __name__ == '__main__':
    main()
