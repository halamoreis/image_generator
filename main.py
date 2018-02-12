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

matplotlib.use('qt4agg')
import matplotlib.pyplot as plt



if __name__ == "__main__":
    generator = ig.ImageGenerator()
    refMod = ReferenceModelDriver()

    numImages = 5
    # Receiving images in the shape 28x28 and 0to1 float format
    subImages = generator.generateSubImage(numImages, 0, 0, reshape=True, convertUChar=False)
    # subImages = generator.generateSubImage(numImages, 0, 0)

    resultRM = refMod.discriminateKNN28x28(np.asarray(subImages))

    print("refMod result:")
    print(resultRM)
    print("   -----   \n\n")

    resultRM = refMod.discriminateSVM28x28(subImages)

    print("refMod result:")
    print(resultRM)
    print("   -----   \n\n")

    subImagesFlat = []
    subImagesUint8 = []

    # Iterating the generated images to aply reference model.
    for i in range(numImages):
        # print("newImg.shape-size")
        # print(subImages[i].shape)
        # print(subImages[i].dtype)
        # print(subImages[i][10])
        plt.imshow(subImages[i], cmap='Greys')
        plt.show()

        # Preparing to aply these images on the full image generator. Converting to uint8.
        subImagesUint8.append(img_as_ubyte(subImages[i]))

        # print(generator.discriminate(subImages[i].reshape(784)))
        # subImagesFlat.append(subImages[i].reshape(784))
        # print(refMod.discriminate(subImages[i]))
        #     Tratando as imagens
        # newImg = cv2.resize(subImages[i], (20, 20)).reshape(400)

        # The RefMod is waiting for 20
        # newImg = cv2.resize(newImg, (20, 20))
        # print("newImg.shape-size")
        # print(newImg.shape)
        # print(newImg.dtype)
        # cv2.imwrite(str(i)+".png", newImg)
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

    resultRM = refMod.discriminateKNN28x28(siarray, uintType=True)

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
    # primaryImage = cv2.imread('./bg/bg2.png')
    #
    # fullImage = generator.generateFullImage(subImagesUint8, primaryImage)
    #
    # plt.imshow(fullImage)
    # plt.show()

