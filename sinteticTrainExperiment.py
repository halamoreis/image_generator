# -*- coding: utf-8 -*-
import cv2 as cv
import numpy as np
import ImageGenerator as ig
import ReferenceModelDriver
import matplotlib

# from pprint import pprint
from skimage import img_as_float
from skimage import img_as_ubyte

from ReferenceModelDriver import ReferenceModelDriver
from tensorflow.examples.tutorials.mnist import input_data

matplotlib.use('qt4agg')
import matplotlib.pyplot as plt


# Initiate kNN, train the data, then test it with test data for k=1
knn = cv.ml.KNearest_create()
SZ = 20
bin_n = 16  # Number of bins

affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR


## [deskew]
def deskew(img):
    m = cv.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SZ * skew], [0, 1, 0]])
    img = cv.warpAffine(img, M, (SZ, SZ), flags=affine_flags)
    return img

## [deskew]

## [hog]

def hog(img):
    gx = cv.Sobel(img, cv.CV_32F, 1, 0)
    gy = cv.Sobel(img, cv.CV_32F, 0, 1)
    mag, ang = cv.cartToPolar(gx, gy)
    bins = np.int32(bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
    bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
    mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
    hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
    hist = np.hstack(hists)  # hist is a 64 bit vector
    return hist

## [hog]




print("--- Initializing CV SVM ref mod ---")

img = cv.imread('digits.png', 0)
if img is None:
    raise Exception("The digits.png image from samples/data is needed!")

cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

# First half is trainData, remaining is testData
train_cells = [i[:50] for i in cells]
test_cells = [i[50:] for i in cells]

# print(type(train_cells))
# print(len(train_cells))
# print(type(train_cells[0]))
# print(len(train_cells[0]))
# print(train_cells[0][0])

# exit(0)

""" Initial boundaries values """

resizeIni = (-0.2, 0.4)
rotateIni = (10, 40)
brightnessIni = (-100, 70)
contrastIni = (1.1, 3)

numImages = 2500

# Generating the test values for the noise generator to be calibrated
resizeParams = np.random.uniform(resizeIni[0], resizeIni[1], numImages)
rotateParams = np.random.randint(rotateIni[0], rotateIni[1], numImages)
brightnessParams = np.random.randint(brightnessIni[0], brightnessIni[1], numImages)
contrastParams = np.random.uniform(contrastIni[0], contrastIni[1], numImages)

gManipulator = ig.ImageGenerator()

# Adding noise to the train set
# alteredImages = [list(gManipulator.addNoiseToLists(row, resizeParams, rotateParams, brightnessParams, contrastParams)) for row in train_cells]
alteredImages = []
for row in train_cells:
    # for item in row:
    #     print("tipo "+str(item.dtype))
    alteredImages.append(list(gManipulator
                         .addNoiseToLists(row, resizeParams, rotateParams, brightnessParams, contrastParams)))
# noisedImages = gManipulator.addNoiseToLists(train_cells, resizeParams, rotateParams, brightnessParams, contrastParams)

print("\n\nGenerated:")
print("Axis x: " + str(len(alteredImages)))
print("Axis y: " + str(len(alteredImages[0])))

# for image in alteredImages:
#     for i in range(2):
#         plt.imshow(image[i*10], cmap='Greys')
#         plt.show()

# train_cells = alteredImages
# train_cells = train_cells + alteredImages
# train_cells = train_cells + test_cells

print("train_cells' new length: "+str(len(train_cells)))
print("Axis x: " + str(len(train_cells)))
print("Axis y: " + str(len(train_cells[0])))
######     Now training      ########################

deskewed = [list(map(deskew, row)) for row in train_cells]
hogdata = [list(map(hog, row)) for row in deskewed]
trainData = np.float32(hogdata).reshape(-1, 64)
responses1 = np.repeat(np.arange(10), 250)[:, np.newaxis]
respAlteredImages = np.repeat(np.arange(10), 250)[:, np.newaxis]

responses = responses1
# responses = np.concatenate((responses1, respAlteredImages), axis=0)

# for i in range(10):
#     print("resp["+str(i)+"] = "+str(responses[i*1000]))

# print("resp["+str(0)+"] = "+str(responses[0]))
# print("resp["+str(250)+"] = "+str(responses[250]))
# print("resp["+str(250+1)+"] = "+str(responses[250+1]))
# print("resp["+str(500)+"] = "+str(responses[500]))
# print("resp["+str(750)+"] = "+str(responses[750]))
# print("resp["+str(1000)+"] = "+str(responses[1000]))
#
# print("\n\n")
#
# print("resp["+str(0)+"] = "+str(responses[0+numImages]))
# print("resp["+str(250)+"] = "+str(responses[250+numImages]))
# print("resp["+str(250+1)+"] = "+str(responses[250+numImages+1]))
# print("resp["+str(500)+"] = "+str(responses[500+numImages]))
# print("resp["+str(750)+"] = "+str(responses[750+numImages]))
# print("resp["+str(1000)+"] = "+str(responses[1000+numImages]))

print("Responses New length: "+str(len(responses)))
print("trainData length: "+str(len(trainData)))

svm = cv.ml.SVM_create()
svm.setKernel(cv.ml.SVM_LINEAR)
svm.setType(cv.ml.SVM_C_SVC)
svm.setC(2.67)
svm.setGamma(5.383)

svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
svm.save('svm_data.dat')

######     Now testing      ########################

deskewed = [list(map(deskew, row)) for row in test_cells]
hogdata = [list(map(hog, row)) for row in deskewed]
testData = np.float32(hogdata).reshape(-1, bin_n * 4)
print("TestData for SVM")
# print(type(testData))
# print(type(testData[0]))
# print(testData[0].shape)
# print(testData[0].dtype)
result = svm.predict(testData)[1]

#######   Check Accuracy   ########################
mask = result == responses1
correct = np.count_nonzero(mask)
print("Accuracy:")
print(correct * 100.0 / result.size)

print("\nSVN Initialized.\n\n")