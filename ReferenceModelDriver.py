import cv2 as cv
import numpy as np
import matplotlib
matplotlib.use('qt4agg')
import matplotlib.pyplot as plt
from skimage import img_as_float
from skimage import img_as_ubyte

class ReferenceModelDriver():
    # Initiate kNN, train the data, then test it with test data for k=1
    knn = cv.ml.KNearest_create()
    SZ = 20
    bin_n = 16  # Number of bins

    affine_flags = cv.WARP_INVERSE_MAP | cv.INTER_LINEAR


    def __init__(self):
        print("Initializing cv ref mod 20x20 0to1 f32")
        with np.load('knn_train-data-20x20-0to1.npz') as data:
            print(data.files)
            train = data['train']
            train_labels = data['train_labels']
        with np.load('knn_test-data-20x20-0to1.npz') as data:
            print(data.files)
            test = data['test']
            test_labels = data['test_labels']

        print("test-type-shape-dtype")
        print(type(test))
        print(test.shape)
        print(test.dtype)

        self.knn.train(train, cv.ml.ROW_SAMPLE, train_labels)
        ret, result, neighbours, dist = self.knn.findNearest(test, k=5)


        matches = result == test_labels
        correct = np.count_nonzero(matches)
        accuracy = correct * 100.0 / result.size
        print("Initial accuracy: ")
        print(accuracy)

        # print("\n\nTesting...")
        # images = []
        # for i in range(5):
        #     print("Displaying "+str(i) + ".png")
        #     img = cv.imread(str(i) + ".png", cv.IMREAD_GRAYSCALE)
        #     # plt.imshow(img, cmap='Greys')
        #     # plt.show()
        #     # img = img_as_float(img.reshape(400))
        #     img = np.float32(img)
        #     img = img.reshape(400)
        #     # img = np.float32(img)
        #     images.append(img)
        #
        # images = np.asarray(images)
        # print("images.shape-dtype")
        # print(images.shape)
        # print(images.dtype)
        # print(type(images))
        # print("images[1].shape-dtype")
        # print(type(images[1]))
        # print(images[1].shape)
        # print(images[1].dtype)
        # print(images[1])
        # ret, result, neighbours, dist = self.knn.findNearest(images, k=5)
        #
        # print(result)
        # print("\n\nFim do teste.\n\n")
        print("\nkNN Initialized.\n\n")

        print("--- Initializing CV SVN ref mod ---")

        img = cv.imread('digits.png', 0)
        if img is None:
            raise Exception("The digits.png image from samples/data is needed!")

        cells = [np.hsplit(row, 100) for row in np.vsplit(img, 50)]

        # First half is trainData, remaining is testData
        train_cells = [i[:50] for i in cells]
        test_cells = [i[50:] for i in cells]

        ######     Now training      ########################

        deskewed = [list(map(self.deskew, row)) for row in train_cells]
        hogdata = [list(map(self.hog, row)) for row in deskewed]
        trainData = np.float32(hogdata).reshape(-1, 64)
        responses = np.repeat(np.arange(10), 250)[:, np.newaxis]

        self.svm = cv.ml.SVM_create()
        self.svm.setKernel(cv.ml.SVM_LINEAR)
        self.svm.setType(cv.ml.SVM_C_SVC)
        self.svm.setC(2.67)
        self.svm.setGamma(5.383)

        self.svm.train(trainData, cv.ml.ROW_SAMPLE, responses)
        self.svm.save('svm_data.dat')

        ######     Now testing      ########################

        deskewed = [list(map(self.deskew, row)) for row in test_cells]
        hogdata = [list(map(self.hog, row)) for row in deskewed]
        testData = np.float32(hogdata).reshape(-1, self.bin_n * 4)
        print("TestData for SVM")
        # print(type(testData))
        # print(type(testData[0]))
        # print(testData[0].shape)
        # print(testData[0].dtype)
        result = self.svm.predict(testData)[1]

        #######   Check Accuracy   ########################
        mask = result == responses
        correct = np.count_nonzero(mask)
        print(correct * 100.0 / result.size)

        print("\nSVN Initialized.\n\n")


    ## [deskew]
    def deskew(self, img):
        m = cv.moments(img)
        if abs(m['mu02']) < 1e-2:
            return img.copy()
        skew = m['mu11'] / m['mu02']
        M = np.float32([[1, skew, -0.5 * self.SZ * skew], [0, 1, 0]])
        img = cv.warpAffine(img, M, (self.SZ, self.SZ), flags=self.affine_flags)
        return img

    ## [deskew]

    ## [hog]

    def hog(self, img):
        gx = cv.Sobel(img, cv.CV_32F, 1, 0)
        gy = cv.Sobel(img, cv.CV_32F, 0, 1)
        mag, ang = cv.cartToPolar(gx, gy)
        bins = np.int32(self.bin_n * ang / (2 * np.pi))  # quantizing binvalues in (0...16)
        bin_cells = bins[:10, :10], bins[10:, :10], bins[:10, 10:], bins[10:, 10:]
        mag_cells = mag[:10, :10], mag[10:, :10], mag[:10, 10:], mag[10:, 10:]
        hists = [np.bincount(b.ravel(), m.ravel(), self.bin_n) for b, m in zip(bin_cells, mag_cells)]
        hist = np.hstack(hists)  # hist is a 64 bit vector
        return hist

    ## [hog]


    """Awaiting for 28x28 images to be discriminated by selected Reference Model."""
    def discriminateKNN28x28(self, images, uintType=False):
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

        if (images[0].shape[0] == 28):
            doImgShape = True
        else:
            doImgShape = False

        newList = []
        # Preventing issues with bad data format.
        # print("images.size " + str(images.shape[0]))
        for i in range(images.shape[0]):
            # Test if already reshaped
            if(not doImgShape):
                imgAux = images[i].reshape(28, 28)
            else:
                imgAux = images[i]
            # print("In KNN discriminator")
            # plt.imshow(imgAux, cmap='Greys')
            # plt.show()
            # Resizing and forcing cast to float32
            if(uintType):
                imgAux = np.float32(imgAux)
            else:
                imgAux = np.float32(img_as_ubyte(imgAux))
            imgAux = cv.resize(imgAux, (20, 20))
            newList.append(imgAux.reshape(400))

        flatResizedArray = np.asarray(newList)

        # print("Info listNP")
        # print(flatResizedArray[0].shape)
        # print(flatResizedArray[0].dtype)
        # print("\n---")
        # print(flatResizedArray[0])
        # print("---\n")
        ret, result, neighbours, dist = self.knn.findNearest(flatResizedArray, k=5)

        return result

    def discriminateSVM28x28(self, images, uintType=False):
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
        # print("images.size " + str(images.shape[0]))
        if (images[0].shape[0] == 28):
            doImgShape = True
        else:
            doImgShape = False

        newList = []
        # Preventing issues with bad data format.
        for i in range(len(images)):
            if (not doImgShape):
                imgAux = images[i].reshape(28, 28)
            else:
                imgAux = images[i]
            # plt.imshow(imgAux, cmap='Greys')
            # plt.show()
            # Resizing and forcing cast to float32
            if(not uintType):
                # print("Converting to uint8")
                imgAux = np.uint8(img_as_ubyte(imgAux))
            imgAux = cv.resize(imgAux, (20, 20))
            newList.append(imgAux)

        # plt.imshow(newList[0], cmap='Greys')
        # plt.show()
        # print("In discriminateSVM method...")
        # print(newList[0])
        otherList = []
        otherList.append(newList)
        deskewed = [list(map(self.deskew, row)) for row in otherList]
        hogdata = [list(map(self.hog, row)) for row in deskewed]
        testData = np.float32(hogdata).reshape(-1, self.bin_n * 4)
        print("Execute SVM")
        # print(type(testData))
        # print(type(testData[0]))
        # print(testData[0].shape)
        # print(testData[0].dtype)
        result = self.svm.predict(testData)[1]

        # flatResizedArray = np.asarray(newList)

        # print("Info listNP")
        # print(flatResizedArray[0].shape)
        # print(flatResizedArray[0].dtype)
        # print("\n---")
        # print(flatResizedArray[0])
        # print("---\n")
        # ret, result, neighbours, dist = self.knn.findNearest(flatResizedArray, k=5)

        return result
