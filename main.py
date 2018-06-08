# -*- coding: utf-8 -*-
import cv2
import numpy as np
import ImageGenerator as ig
import ReferenceModelDriver
import matplotlib
import time

# from pprint import pprint
from skimage import img_as_float
from skimage import img_as_ubyte

from ReferenceModelDriver import ReferenceModelDriver

matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

# 2018-03-22 17:42:26.700837: I tensorflow/core/platform/cpu_feature_guard.cc:137] Your CPU supports instructions that this TensorFlow binary was not compiled to use: SSE4.1 SSE4.2 AVX AVX2 FMA

if __name__ == "__main__":
    generator = ig.ImageGenerator()
    refMod = ReferenceModelDriver()

    # UniKernel test
    numTestImages = 1000
    mnist = input_data.read_data_sets("./MNIST_data/")
    testSet = mnist.test.images[:numTestImages]

    # print(type(testSet))
    # print(testSet.shape)

    sizeSqrImg = 28
    # numLinesSqr = int(math.sqrt(numTestImages)) * sizeSqrImg
    # plt.imshow(testSet.reshape((16, 28, 28), order='C').reshape(28*8, 28*2), cmap='Greys')
    # plt.show()
    #
    # plt.imshow(testSet.reshape((numLinesSqr, numLinesSqr), order='C'), cmap='Greys')
    # plt.show()

    # Opening the BG image.
    # primaryImage = cv2.imread('./bg/bg2.png')

    imgListUint8_28x28 = []
    for img in testSet:
        imgListUint8_28x28.append(img_as_ubyte(img.reshape(28, 28)))
    #
    # fullImage = generator.generateFullImage(imgListUint8_28x28, primaryImage)
    #
    # plt.imshow(fullImage)
    # plt.show()

    images_array = np.asarray(imgListUint8_28x28)

    # for img in images_array:
    #     plt.imshow(img, cmap="Greys")
    #     plt.show()

    """ Initial boundaries values """

    resizeIni = (-0.3, 0.8)
    rotateIni = (5, 55)
    brightnessIni = (-100, 70)
    contrastIni = (1.1, 2)

    # Generating the test values for the noise generator to be calibrated
    resizeParams = np.linspace(resizeIni[0], resizeIni[1], numTestImages)
    rotateParams = np.random.uniform(rotateIni[0], rotateIni[1], numTestImages)
    rotateParams = np.float32(rotateParams)
    brightnessParams = np.random.randint(brightnessIni[0], brightnessIni[1], numTestImages)
    # brightnessParams = np.int16(brightnessParams)
    contrastParams = np.random.uniform(contrastIni[0], contrastIni[1], numTestImages)
    # print(contrastParams.shape)
    # print(contrastParams.size)
    # print type(contrastParams)
    # print(contrastParams)

    contrastParams = np.float32(contrastParams)

    # print("\n")
    # print(contrastParams.shape)
    # print(contrastParams.size)
    # print type(contrastParams)
    # print(contrastParams)

    start = time.time()
    alteredGPUImages = generator.addNoiseGPU(images_array, resizeParams, rotateParams, brightnessParams, contrastParams)
    end = time.time()
    print("GPU time: " + str(end - start))


    """ TEMP """
    start = time.time()
    alteredCPUImages = []
    for i in range(numTestImages):
        fakeList = []
        fakeList.append(images_array[i].reshape(28, 28))
        # plt.imshow(imageSetResize[i].reshape(28, 28), cmap='Greys')
        # plt.show()
        newImage = generator.addNoise(fakeList, 0, rotateParams[i], brightnessParams[i], np.float64(contrastParams[i]))
        alteredCPUImages.append(newImage[0])

    end = time.time()
    print("CPU time: " + str(end - start))

    start = time.time()
    alteredCPUListImages = generator.addNoiseToLists(images_array, resizeParams, rotateParams, brightnessParams,
                                                     contrastParams)
    end = time.time()
    print("CPU in-list time: " + str(end - start))

    print("\nBrightness Params:\n" + str(brightnessParams[0]) + " " + str(brightnessParams[1]) + " " + str(
        brightnessParams[2]))
    print("\nContrast Params:\n" + str(contrastParams[0]) + " " + str(contrastParams[1]) + " " + str(contrastParams[2]))
    print("\nRotate Params:\n" + str(rotateParams[0]) + " " + str(rotateParams[1]) + " " + str(rotateParams[2]))
    print("\n")

    # print("Printing Original sample.")
    # for i in range(3):
    #     plt.imshow(images_array[i], cmap="Greys")
    #     plt.show()

    print("Printing CPU sample.")
    for i in range(3):
        plt.imshow(alteredCPUImages[i], cmap="Greys")
        plt.show()
        # print(alteredCPUImages[i])

    print("\n\n\nPrinting GPU sample.")
    for i in range(3):
        plt.imshow(alteredGPUImages[i], cmap="Greys")
        plt.show()
        print(alteredGPUImages[i][0])
        print(alteredGPUImages[i].shape)
        print type(alteredGPUImages[i])

    print("Printing CPU in-list sample.")
    for i in range(3):
        plt.imshow(alteredCPUListImages[i], cmap="Greys")
        plt.show()
        # print(alteredCPUImages[i])

    # for img in alteredImages:
    #     plt.imshow(img, cmap="Greys")
    #     plt.show()

    print("\nExiting...")
    exit(0)



    numImages = 100
    # Receiving images in the shape 28x28 and 0to1 float format
    subImages, reliability = generator.generateSubImageWPrejudice(numImages, 0, 0, reshape=False, convertUChar=False)
    # subImages = generator.generateSubImage(numImages, 0, 0)

    print(subImages.size)
    print(len(subImages))

    resultKNN = refMod.discriminateKNN28x28(np.asarray(subImages))

    print("refMod result:")
    print(resultKNN)
    print("   -----   \n\n")

    resultSVM = refMod.discriminateSVM28x28(subImages)
    """
    print("refMod result:")
    print(resultSVM)
    print("   -----   \n\n")

    print("Reliability:")
    print(reliability)
    print("   -----   \n\n")
    """

    # resultCNNGen, rVals = generator.cnnDiscriminator(np.asarray(subImages).reshape(-1, 784))
    resultCNNGen, rVals = generator.cnnDiscriminator(subImages)

    print("CNNGen result:")
    # print(resultCNNGen)
    for i in range(len(resultCNNGen)):
        print("Class: "+str(resultCNNGen[i])+" - R: "+str(np.amax(rVals[i])))
    print("   -----   \n\n")

    subImagesFlat = []
    subImagesUint8 = []

    # Iterating the generated images to aply reference model.
    for i in range(len(subImages)):
        # print("newImg.shape-size")
        # print(subImages[i].shape)
        # print(subImages[i].dtype)
        # print(subImages[i][10])

        # Preparing to aply these images on the full image generator. Reshaping to 28x28 and Converting to uint8.
        imgTemp = subImages[i].reshape(28, 28)
        subImagesUint8.append(img_as_ubyte(subImages[i]))

        # print(subImages[i].dtype)
        # print(type(subImages[i]))
        # print("Discriminator says:")
        # print(generator.discriminate([subImages[i].reshape(784)]))
        # subImagesFlat.append(subImages[i].reshape(784))
        # print(refMod.discriminate(subImages[i]))
        #     Tratando as imagens
        # newImg = cv2.resize(subImages[i], (20, 20)).reshape(400)

        # plt.imshow(imgTemp, cmap='Greys')
        # plt.show()

        cv2.imwrite("syn-img/"+str(i).zfill(3)+"_P_"+str(np.amax(rVals[i]))+"_"+str(resultCNNGen[i])+"-"+str(resultSVM[i][0])+".png", img_as_ubyte(imgTemp))


        # The RefMod is waiting for 20
        # newImg = cv2.resize(newImg, (20, 20))
        # print("newImg.shape-size")
        # print(newImg.shape)
        # print(newImg.dtype)

        # bkpStr = "img_"+str(i)+".npz"
        # print("Writing "+bkpStr)
        # np.savez(bkpStr, newImg=newImg)

        # plt.imshow(newImg, cmap='Greys')
        # plt.show()

        # newImg = np.float32(newImg.reshape(400))
        # print("\n\nnewImg.shape-dtype")
        # print(newImg.shape)
        # print(newImg.dtype)

        # newImg = subImages[i].reshape(784)
        # subImagesFlat.append(newImg)
    # subImagesFlat = np.asarray(subImagesFlat)
    # print("subImagesFlat.dtype")
    # print(subImagesFlat.dtype)

    # print("Agora visualizando a figura completa.")

    # plt.imshow(subImagesUint8, cmap='Greys')
    # plt.show()

    print("Encerrando...")
    exit(0)

    siarray = np.asarray(subImagesUint8)

    """Prepare subImages adding noise."""
    subImagesUint8 = generator.addNoise(siarray, -0.4, 0, 0, 0)
    # newList = []

    resultRM = refMod.discriminateKNN28x28(siarray, uintType=True)

    print("refMod result:")
    print(resultRM)
    print("   -----   \n\n")

    # for i in range(numImages):
        # plt.imshow(subImagesUint8[i], cmap='Greys')
        # plt.show()
        # newList.append(subImagesUint8[i])

    # newArray = np.asarray(newList)
    siarray = np.asarray(subImagesUint8)


    """Prepare subImages adding more noise."""
    subImagesUint8 = generator.addNoise(siarray, 0, 35, 0, 0)

    resultRM = refMod.discriminateSVM28x28(siarray, uintType=True)

    print("refMod result:")
    print(resultRM)
    print("   -----   \n\n")

    # for i in range(numImages):
        # print("newImg.shape-size")
        # print(subImages[i].shape)
        # print(subImages[i].dtype)
        # print(subImages[i][10])
        # plt.imshow(subImagesUint8[i], cmap='Greys')
        # plt.show()

    """Prepare subImages adding more noise."""
    subImagesUint8 = generator.addNoise(subImagesUint8, 0, 0, 20, 1.5)

    resultRM = refMod.discriminateKNN28x28(np.asarray(subImagesUint8), uintType=True)

    print("refMod result:")
    print(resultRM)
    print("   -----   \n\n")

    for i in range(numImages):
        # print("newImg.shape-size")
        # print(subImages[i].shape)
        # print(subImages[i].dtype)
        # print(subImages[i][10])
        plt.imshow(subImagesUint8[i], cmap='Greys')
        plt.show()

    # Opening the BG image.
    primaryImage = cv2.imread('./bg/bg2.png')
    #
    fullImage = generator.generateFullImage(subImagesUint8, primaryImage)
    #
    plt.imshow(fullImage)
    plt.show()

