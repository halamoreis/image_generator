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
    # subImages = generator.generateSubImage(numImages, 0, 0, reshape=False, convertUChar=False)
    subImages = generator.generateSubImage(numImages, 0, 0)

    # print(subImages.__class__)

    subImagesFlat = []

    for i in range(numImages):
        newImg = img_as_ubyte(subImages[i].reshape(28, 28))
        plt.imshow(newImg, cmap='Greys')
        plt.show()
        # print(generator.discriminate(subImages[i].reshape(784)))
        # subImagesFlat.append(subImages[i].reshape(784))
        # print(refMod.discriminate(subImages[i]))
        #     Tratando as imagens
        # newImg = cv2.resize(subImages[i], (20, 20)).reshape(400)
        newImg = cv2.resize(newImg, (20, 20))
        # print("newImg.shape-size")
        # print(newImg.shape)
        # print(newImg.dtype)
        # cv2.imwrite(str(i)+".png", newImg)
        # bkpStr = "img_"+str(i)+".npz"
        # print("Writing "+bkpStr)
        # np.savez(bkpStr, newImg=newImg)

        # plt.imshow(newImg, cmap='Greys')
        # plt.show()

        newImg = np.float32(newImg.reshape(400))
        # print("\n\nnewImg.shape-dtype")
        # print(newImg.shape)
        # print(newImg.dtype)

        # newImg = subImages[i].reshape(784)
        subImagesFlat.append(newImg)
    # subImagesFlat = np.asarray(subImagesFlat)
    print("subImagesFlat.dtype")
    # print(subImagesFlat.dtype)
    result = refMod.discriminate(np.asarray(subImagesFlat))

    print("refMod result:")
    print(result)
    print(result[0][0])
    print(result[1][0])
    print(result[2][0])
    print(result.shape)
    print(result.dtype)
    print(type(result))
    print("   -----   \n\n")

    # Opening the BG image.
    primaryImage = cv2.imread('./bg/bg2.png')

    fullImage = generator.generateFullImage(subImages, primaryImage)

    plt.imshow(fullImage)
    plt.show()

