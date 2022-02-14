import gzip
import numpy as np
from PIL import Image
import PIL.ImageOps

image_size = 28
NUM_IMAGES = 60000
DIM = 28, 28
##################################################### loading #####################################################


def train_loader(num_images=NUM_IMAGES):
    train_images = gzip.open("./data/train-images-idx3-ubyte.gz", "r")
    train_labels = gzip.open("./data/train-labels-idx1-ubyte.gz", "r")

    train_images.read(16)
    buf = train_images.read(image_size * image_size * num_images)
    data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
    images = data.reshape(num_images, image_size * image_size)

    train_labels.read(8)
    buf = train_labels.read()
    targets = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    targets = targets[:num_images]
    return images, targets


def load_single_image(image_file="./data/6.jpg"):
    DIM = 28, 28

    image = Image.open(image_file)
    image = PIL.ImageOps.invert(image)
    image = image.convert("L")
    img_file = image.resize(DIM)
    loaded_img = np.array(img_file)

    return loaded_img.reshape(1, -1)
