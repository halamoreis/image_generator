import cv2 as cv
import numpy as np



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

        print("test")
        print(type(test))
        print(test.shape)
        self.knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
        ret, result, neighbours, dist = self.knn.findNearest(test, k=5)

        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        print("Initial accuracy: ")
        print(accuracy)

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
