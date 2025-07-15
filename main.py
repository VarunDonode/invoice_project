import argparse
from model import VisionModel
from extractor import InvoiceExtractor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str)
    args = parser.parse_args()

    model = VisionModel()
    extractor = InvoiceExtractor(model)

    result = extractor.extract(args.img_path)
    print(result)

if __name__ == '__main__':
    main()
