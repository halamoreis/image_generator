# -*- coding: utf-8 -*-
import cv2
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



if __name__ == "__main__":
    generator = ig.ImageGenerator()
    refMod = ReferenceModelDriver()

    numImages = 500
    numCalibrationImages = 500
    # Receiving images in the shape 28x28 and 0to1 float format
    # subImages = generator.generateSubImage(numImages, 0, 0, reshape=True, convertUChar=False)

    # Loading the MNIST database
    mnist = input_data.read_data_sets("./MNIST_data/")

    # print(type(mnist.test.labels))
    # print(type(mnist.test.labels[0]))
    # print("test label[0]")
    # print(mnist.test.labels[0])
    # print(len(mnist.test.labels))
    # print("test image")
    # print(type(mnist.test.images[0]))
    # print(mnist.test.images[0].reshape((28, 28)))

    # Separe  in sub-sets to calibrate the suitable noise
    # calibrateSet = mnist.train.images[:numImages]
    # calibrateSetLabels = mnist.train.labels[:numImages]
    calibrateSet = mnist.train.images[:numCalibrationImages]
    calibrateSetLabels = mnist.train.labels[:numCalibrationImages]
    # print("test image")
    # print(calibrateSetLabels)
    # exit(0)
    # print(type(calibrateSet))
    # print(type(calibrateSet[0]))
    # print(calibrateSet[0])



    # Discriminating with the reference models
    # for i in range(len(calibrateSet)):
    resultKNN = refMod.discriminateKNN28x28(calibrateSet)
    resultSVM = refMod.discriminateSVM28x28(calibrateSet)

    newList = []
    newList.append(calibrateSetLabels)
    calibrateSetLabels = np.asarray(newList)

    resultKNN = np.asarray(resultKNN, dtype=np.int)
    resultKNN = resultKNN.reshape(numImages)
    resultSVM = np.asarray(resultSVM, dtype=np.int)
    resultSVM = resultSVM.reshape(numImages)
    calibrateSetLabels = calibrateSetLabels.reshape(numImages)

    maskKNN = resultKNN == calibrateSetLabels
    maskSVM = resultSVM == calibrateSetLabels

    print("\nKNN Accuracy:")
    correct = np.count_nonzero(maskKNN)
    print(correct * 100.0 / resultKNN.size)

    print("\nSVM Accuracy:")
    correct = np.count_nonzero(maskSVM)
    print(correct * 100.0 / maskSVM.size)

    # print("test labels")
    # print(calibrateSetLabels.shape)
    # print(len(calibrateSetLabels))
    # print(type(calibrateSetLabels))
    # print(type(calibrateSetLabels[0]))
    # print(calibrateSetLabels)
    # print(calibrateSetLabels[0])
    # print(calibrateSetLabels[0][10])

    # print("results")
    # print(resultKNN.shape)
    # print(len(resultKNN))
    # print(type(resultKNN))
    # print(type(resultKNN[0]))
    # print(resultKNN)
    # print(resultKNN[0])
    # print(resultKNN[10][0])

    # print("results SVM")
    # print(resultSVM.shape)
    # print(len(resultSVM))
    # print(type(resultSVM))
    # print(type(resultSVM[0]))
    # print(resultSVM)

    # if(calibrateSetLabels[0][10] == resultKNN[0][10][0]):
    # print("Primeiro resultado correto")

    # print("\n\nMAsKresults")
    # print(len(maskKNN))
    # print(type(maskKNN))
    # print(type(maskKNN[0]))
    # print(maskKNN)
    # print(maskKNN[0])



    """ Initial boundaries values """

    resizeIni = (-0.7, 0.8)
    rotateIni = (5, 60)
    brightnessIni = (-100, 70)
    contrastIni = (1.1, 3)

    # Generating the test values for the noise generator to be calibrated
    resizeParams = np.linspace(resizeIni[0], resizeIni[1], numCalibrationImages)
    rotateParams = np.linspace(rotateIni[0], rotateIni[1], numCalibrationImages)
    brightnessParams = np.linspace(brightnessIni[0], brightnessIni[1], numCalibrationImages)
    contrastParams = np.linspace(contrastIni[0], contrastIni[1], numCalibrationImages)

    calibrateSet28x28 = []
    # Reshaping image set
    for i in range(numCalibrationImages):
        calibrateSet28x28.append(calibrateSet[i].reshape(28, 28))


    """ Main calibration loops"""
    print("Calibrating resize param...")
    # print(resizeParams)
    # Calibrating resize params

    newImagesList = []
    for i in range(numCalibrationImages):
        fakeList = []
        fakeList.append(calibrateSet28x28[i])
        newImage = generator.addNoise(fakeList, resizeParams[i], 0, 0, 0)
        newImagesList.append(newImage)
    calibrateSetArray = np.asarray(newImagesList)
    resultKNN = refMod.discriminateKNN28x28(calibrateSetArray)
    resultSVM = refMod.discriminateSVM28x28(calibrateSetArray)

    resultKNN = np.uint8(resultKNN.reshape(numCalibrationImages))
    resultSVM = np.uint8(resultSVM.reshape(numCalibrationImages))

    # print(resizeParams)
    # print(resultKNN)
    # print(resultSVM)
    # print(calibrateSetLabels[:numCalibrationImages])

    boolMask = resultSVM == calibrateSetLabels
    correct = np.count_nonzero(boolMask)

    newMin = resizeIni[0]
    newMax = resizeIni[1]
    searchingMin = True
    searchingMax = False
    for i in range(numCalibrationImages):
        if(boolMask[i]):
            if(searchingMin):
                newMin = resizeParams[i]
                searchingMin = False
                searchingMax = True
            elif(searchingMax):
                newMax = resizeParams[i]

    # Update the new calibrated boundaries
    resizeCalibrated = (newMin, newMax)
    print("New boundaries for resize:")
    # print(resizeCalibrated)

    print("\n\nCalibrating rotate param...")
    # Calibrating resize params
    newRotateMin = rotateIni[0]
    newRotateMax = rotateIni[1]
    newImagesList = []
    for i in range(numCalibrationImages):
        fakeList = []
        fakeList.append(calibrateSet28x28[i])
        newImage = generator.addNoise(fakeList, 0, rotateParams[i], 0, 0)
        newImagesList.append(newImage)
    calibrateSetArray = np.asarray(newImagesList)
    resultKNN = refMod.discriminateKNN28x28(calibrateSetArray)
    resultSVM = refMod.discriminateSVM28x28(calibrateSetArray)

    resultKNN = np.uint8(resultKNN.reshape(numCalibrationImages))
    resultSVM = np.uint8(resultSVM.reshape(numCalibrationImages))

    # print(rotateParams)
    # print(resultKNN)
    # print(resultSVM)
    # print(calibrateSetLabels[:numCalibrationImages])

    boolMask = resultSVM == calibrateSetLabels
    correct = np.count_nonzero(boolMask)

    newMin = rotateIni[0]
    newMax = rotateIni[1]
    searchingMin = True
    searchingMax = False
    for i in range(numCalibrationImages):
        if (boolMask[i]):
            if (searchingMin):
                newMin = rotateParams[i]
                searchingMin = False
                searchingMax = True
            elif (searchingMax):
                newMax = rotateParams[i]

    # Update the new calibrated boundaries
    rotateCalibrated = (newMin, newMax)
    print("New boundaries for rotate:")
    print(rotateCalibrated)

    print("\n\nCalibrating brightness and contrast params...")
    # Calibrating resize params
    # newRotateMin = rotateIni[0]
    # newRotateMax = rotateIni[1]
    newImagesList = []
    for i in range(numCalibrationImages):
        fakeList = []
        fakeList.append(calibrateSet28x28[i])
        newImage = generator.addNoise(fakeList, 0, 0, brightnessParams[i], contrastParams[i])
        newImagesList.append(newImage)
    calibrateSetArray = np.asarray(newImagesList)
    resultKNN = refMod.discriminateKNN28x28(calibrateSetArray)
    resultSVM = refMod.discriminateSVM28x28(calibrateSetArray)

    resultKNN = np.uint8(resultKNN.reshape(numCalibrationImages))
    resultSVM = np.uint8(resultSVM.reshape(numCalibrationImages))

    # print(brightnessParams)
    # print(contrastParams)
    # print(resultKNN)
    # print(resultSVM)
    # print(calibrateSetLabels)

    boolMask = resultSVM == calibrateSetLabels
    correct = np.count_nonzero(boolMask)
    # print(boolMask)
    print(correct * 100.0 / numCalibrationImages)

    newMinb = brightnessIni[0]
    newMinc = contrastIni[0]
    newMaxb = brightnessIni[1]
    newMaxc = contrastIni[1]
    searchingMin = True
    searchingMax = False
    for i in range(numCalibrationImages):
        if (boolMask[i]):
            if (searchingMin):
                newMinb = brightnessParams[i]
                newMinc = contrastParams[i]
                searchingMin = False
                searchingMax = True
            elif (searchingMax):
                newMaxb = brightnessParams[i]
                newMaxc = contrastParams[i]

    # Update the new calibrated boundaries
    brightCalibrated = (newMinb, newMaxb)
    contrastCalibrated = (newMinc, newMaxc)
    print("New boundaries for brightness and contrast:")
    print(brightCalibrated)
    print(contrastCalibrated)




    """"""
    ####################################################
    #
    ####################################################
    """"""

    # Getting the image test set and the corresponding labels
    numTestImages = 200
    numResizedImages = int(numTestImages / 4)
    numRotatedImages = int(numTestImages / 4)
    numBCImages = int(numTestImages / 4)
    numResizedRotatedImages = int(numTestImages / 4)

    testSet = mnist.test.images[:numTestImages]
    testLabels = mnist.test.labels[:numTestImages]

    # Separating in sub-sets
    imageSetResize = testSet[:numResizedImages]
    labelsResize = testLabels[:numResizedImages]

    indexIni = numResizedImages
    indexEnd = numResizedImages+numRotatedImages
    imageSetRotate = testSet[indexIni:indexEnd]
    labelsRotate = testLabels[indexIni:indexEnd]

    indexIni = indexEnd
    indexEnd += numBCImages
    imageSetBC = testSet[indexIni:indexEnd]
    labelsBC = testLabels[indexIni:indexEnd]

    indexIni = indexEnd
    indexEnd += numResizedRotatedImages
    imageSetRR = testSet[indexIni:indexEnd]
    labelsRR = testLabels[indexIni:indexEnd]

    """ Generating new values for noise generation to run the verification trial """
    resizeParams = np.linspace(resizeCalibrated[0], resizeCalibrated[1], numResizedImages)
    rotateParams = np.linspace(rotateCalibrated[0], rotateCalibrated[1], numRotatedImages)
    brightnessParams = np.linspace(brightCalibrated[0], brightCalibrated[1], numBCImages)
    contrastParams = np.linspace(contrastCalibrated[0], contrastCalibrated[1], numBCImages)

    """ Execute the Resize test """
    print("\n\nExecute the Resize test:")
    newImagesList = []
    for i in range(numResizedImages):
        fakeList = []
        fakeList.append(imageSetResize[i].reshape(28, 28))
        newImage = generator.addNoise(fakeList, resizeParams[i], 0, 0, 0)
        newImagesList.append(newImage)
    imageSetArray = np.asarray(newImagesList)
    resultKNN = refMod.discriminateKNN28x28(imageSetArray)
    resultSVM = refMod.discriminateSVM28x28(imageSetArray)

    resultKNN = np.uint8(resultKNN.reshape(numResizedImages))
    resultSVM = np.uint8(resultSVM.reshape(numResizedImages))

    print("Accuracy for Resize test in OpenCV KNN:")
    boolMask = resultKNN == labelsRotate
    correct = np.count_nonzero(boolMask)
    print(correct * 100.0 / numResizedImages)

    print("Accuracy for Resize test in OpenCV SVM:")
    boolMask = resultSVM == labelsRotate
    correct = np.count_nonzero(boolMask)
    print(correct * 100.0 / numResizedImages)


    """ Execute the Rotate test """
    print("\n\nExecute the Rotate test:")
    newImagesList = []
    for i in range(numRotatedImages):
        fakeList = []
        fakeList.append(imageSetRotate[i].reshape(28, 28))
        newImage = generator.addNoise(fakeList, 0, rotateParams[i], 0, 0)
        newImagesList.append(newImage)
    rotatedSetArray = np.asarray(newImagesList)
    resultKNN = refMod.discriminateKNN28x28(rotatedSetArray)
    resultSVM = refMod.discriminateSVM28x28(rotatedSetArray)

    resultKNN = np.uint8(resultKNN.reshape(numRotatedImages))
    resultSVM = np.uint8(resultSVM.reshape(numRotatedImages))

    print("Accuracy for Rotate test in OpenCV KNN:")
    boolMask = resultKNN == labelsRotate
    correct = np.count_nonzero(boolMask)
    print(correct * 100.0 / numRotatedImages)

    print("Accuracy for Rotate test in OpenCV SVM:")
    boolMask = resultSVM == labelsRotate
    correct = np.count_nonzero(boolMask)
    print(correct * 100.0 / numRotatedImages)
    # print(resultSVM)
    # print(labelsRotate)
    # print(boolMask)




    # testSet28x28 = []
    # # Reshaping image set
    # for i in range(numTestImages):
    #     testSet28x28.append(testSet[i].reshape(28, 28))



