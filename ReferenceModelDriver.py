import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from skimage import img_as_float

class ReferenceModelDriver():
    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv.ml.KNearest_create()

    def __init__(self):
        print("Initializing cv ref mod")
        with np.load('knn_train-data.npz') as data:
            print(data.files)
            train = data['train']
            train_labels = data['train_labels']
        with np.load('knn_test-data.npz') as data:
            print(data.files)
            test = data['test']
            test_labels = data['test_labels']

        print("test-type-shape-dtype")
        print(type(test))
        print(test.shape)
        print(test.dtype)
        print("test[200]-type-shape-dtype")
        print(type(test[200]))
        print(test[200].shape)
        print(test[200].dtype)
        print(test[200])

        self.knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
        ret, result, neighbours, dist = self.knn.findNearest(test, k=5)

        print("Results")
        print(result.shape)
        print(result.dtype)
        print(type(result))
        print(result)
        print(result[0][0])
        print(result[1][0])
        print(result[2000][0])

        # plt.imshow(test[1].reshape(20, 20), cmap='Greys')
        # plt.show()
        # plt.imshow(test[2000].reshape(20, 20), cmap='Greys')
        # plt.show()

        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        print("Initial accuracy: ")
        print(accuracy)

        print("\n\nTesting...")
        images = []
        for i in range(5):
            print("Displaying "+str(i) + ".png")
            img = cv.imread(str(i) + ".png", cv.IMREAD_GRAYSCALE)
            # plt.imshow(img, cmap='Greys')
            # plt.show()
            # img = img_as_float(img.reshape(400))
            img = np.float32(img)
            img = img.reshape(400)
            # img = np.float32(img)
            images.append(img)

        images = np.asarray(images)
        print("images.shape-dtype")
        print(images.shape)
        print(images.dtype)
        print(type(images))
        print("images[1].shape-dtype")
        print(type(images[1]))
        print(images[1].shape)
        print(images[1].dtype)
        print(images[1])
        ret, result, neighbours, dist = self.knn.findNearest(images, k=5)

        print(result)
        print("\n\nFim do teste.\n\n")

    def discriminate(self, images):
        # print(image.shape)
        # image = image.reshape(784)
        # print(image.shape)
        #
        # # Resizing image
        # image = cv.resize(image, (20, 20))
        #
        # print("New shape")
        # print(image.shape)
        # image = image.reshape(400)
        # print(image.shape)

        # images = image

        ret, result, neighbours, dist = self.knn.findNearest(images, k=5)

        return result
