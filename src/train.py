from sklearn.model_selection import train_test_split
from sklearn import svm

from src.data.load import train_loader


image_size = 28
num_images = 60000

##################################################### loading #####################################################

images, targets = train_loader(num_images=num_images)
targets = targets[:num_images]

##################################################### training #####################################################

train_images, test_images, train_labels, test_labels = train_test_split(
    images, targets, train_size=0.8, random_state=0
)
print(len(train_images))

clf = svm.SVC()
clf.fit(train_images, train_labels)
clf.score(test_images, test_labels)
print(clf.score(test_images, test_labels))
