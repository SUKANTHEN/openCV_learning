import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

img = cv.imread('digits.png')
gray = cv.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Now we split the image to 5000 cells, each of 20x20 size
cells = [np.hsplit(row,100) for row in np.vsplit(gray,50)]
x = np.array(cells)

# Now we prepare train_data and test_data.
train = x[:,:50].reshape(-1,400).astype(np.float32) # Size = (2500,400)
test = x[:,50:100].reshape(-1,400).astype(np.float32) # Size = (2500,400)

# Create labels for train and test data
k = np.arange(10)
train_labels = np.repeat(k,250)[:,np.newaxis]
test_ labels = train.copy()

# Initiate kNN, train the data, then test it with test data for k=1
knn = cv.KNearest()
knn.train(train,train_labels)
ret,res,neighbours,dist = knn.find_nearest(test,k=5)

matches = res==test_labels
correct = np.count_nonzero(matches)
accuracy = correct*100/res.size
