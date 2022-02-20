import argparse
import pickle
import yaml
import numpy as np
from sklearn import svm
from PIL import Image
import PIL.ImageOps
import matplotlib.pyplot as plt

DIM = 28, 28


def predict_one_shot(model, image_path):
    """Run the inference on the test set and writes the output on a csv file

    Args:
        model (pkl): model to use
        image_path (str): path to image
    """

    # Load test data
    # loaded_img = load_single_image()

    image = Image.open(image_path)

    image = PIL.ImageOps.invert(image)
    image = image.convert("L")
    img_file = image.resize(DIM)
    loaded_img = np.array(img_file)

    prediction = model.predict(loaded_img.reshape(1, -1))
    return prediction
