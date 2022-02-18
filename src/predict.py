import argparse
import pickle
import yaml
import numpy as np
from sklearn import svm
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt

from data.load import load_single_image

DIM = 28, 28


def predict_one_shot(cfg):
    """Run the inference on the test set and writes the output on a csv file

    Args:
        cfg (dict): configuration
    """

    # Load test data
    # loaded_img = load_single_image()
    image = "../data/6.jpg"
    image = Image.open(image)

    image = PIL.ImageOps.invert(image)
    image = image.convert("L")
    img_file = image.resize(DIM)
    loaded_img = np.array(img_file)

    # Load model
    model = pickle.load(open(cfg["TEST"]["PATH_TO_MODEL"], "rb"))

    prediction = model.predict(loaded_img.reshape(1, -1))

    img = img_file
    plt.imshow(img, cmap="binary")
    plt.title(prediction)
    plt.show()


if __name__ == "__main__":
    # Init the parser;
    inference_parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter
    )

    # Add path to the config file to the command line arguments;
    inference_parser.add_argument(
        "--path_to_config",
        type=str,
        required=True,
        default="./config.yaml",
        help="path to config file.",
    )
    args = inference_parser.parse_args()

    # Load config file
    with open(args.path_to_config, "r") as ymlfile:
        config_file = yaml.load(ymlfile, Loader=yaml.Loader)

    # Run prediction
    predict_one_shot(cfg=config_file)
