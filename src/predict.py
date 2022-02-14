import gzip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
from sklearn import svm
from PIL import Image
import PIL.ImageOps

##################################################### prediction #####################################################
i = np.random.randint(100)


prediction = clf.predict(test_images[i].reshape(1, -1))


img = test_images[
    i,
].reshape((28, 28))
plt.imshow(img, cmap="binary")
plt.title(prediction)
plt.show()

##################################################### loading #####################################################


DIM = 28, 28  # 256, 256

image = "./data/6.jpg"
image = Image.open(image)


image = PIL.ImageOps.invert(image)
image = image.convert("L")
img_file = image.resize(DIM)
loaded_img = np.array(img_file)
prediction = clf.predict(loaded_img.reshape(1, -1))


img = img_file
plt.imshow(img, cmap="binary")
plt.title(prediction)
plt.show()
