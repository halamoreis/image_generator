# -*- coding: utf-8 -*-
import cv2
import numpy as np
import ImageGenerator as ig
import ReferenceModelDriver
import matplotlib

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
        # plt.imshow(subImages[i].reshape(28, 28), cmap='Greys')
        plt.imshow(subImages[i], cmap='Greys')
        plt.show()
        # print(generator.discriminate(subImages[i].reshape(784)))
        # subImagesFlat.append(subImages[i].reshape(784))
        # print(refMod.discriminate(subImages[i]))
        #     Tratando as imagens
        # newImg = cv2.resize(subImages[i], (20, 20)).reshape(400)
        newImg = cv2.resize(subImages[i], (20, 20))

        # plt.imshow(newImg, cmap='Greys')
        # plt.show()

        newImg = newImg.reshape(400)

        # newImg = subImages[i].reshape(784)
        subImagesFlat.append(newImg)

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

