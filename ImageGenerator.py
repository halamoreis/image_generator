# -*- coding: utf-8 -*-
import tensorflow as tf
import matplotlib

matplotlib.use('qt4agg')
from matplotlib import pyplot as plt

import pycuda.driver as cuda
from pycuda.compiler import SourceModule
import pycuda.autoinit

import numpy as np
import cv2 as cv

from skimage import img_as_ubyte
from tensorflow.examples.tutorials.mnist import input_data


class ImageGenerator:
    # Default values
    MNIST_DIR = "./MNIST_data/"
    TRAINED_MODEL_DIR = "/home/hlm/workspace/python/DL/TF/models/500_epoch_model.ckpt"
    resolution = 28
    dataset = "MNIST"



    # Kernel for combining bg and fg images
    kernel1_code = """
      __global__ void printPart(unsigned char *bg, unsigned char *part, long int *offsets, unsigned char *lineSize)
      {
            int imgOffset = 100, indexOffset = 28*28;
                                                // Number of offsets
            int offsetx = offsets[blockIdx.x];
            int offsety = offsets[blockIdx.x + 4];
            int id = (threadIdx.x + offsetx) + ((threadIdx.y+offsety) * 1024);
            int valPart = part[(threadIdx.x + threadIdx.y * blockDim.y) + (indexOffset * blockIdx.x)];

            valPart != 0 ? bg[id*3] = (bg[id] * 0.3) + (valPart * 0.7) + 0: 0;
            valPart != 0 ? bg[(id*3)+1] = (bg[id] * 0.3) + (valPart * 0.7) + 0: 0;
            valPart != 0 ? bg[(id*3)+2] = (bg[id] * 0.3) + (valPart * 0.7) + 0: 0;




            // bg[id] = blockDim.x-15;
            //bg[id] = threadIdx.x + (threadIdx.y * blockDim.y);
            //part[id] = blockDim.y;
      }
      """

    kernelBright_code = """
      __global__ void printPart(unsigned char *bg, unsigned char *part, long int *offsets, unsigned char *lineSize)
      {
            int imgOffset = 100, indexOffset = 28*28;
                                                // Number of offsets
            int offsetx = offsets[blockIdx.x];
            int offsety = offsets[blockIdx.x + 4];
            int id = (threadIdx.x + offsetx) + ((threadIdx.y+offsety) * 1024);
            int valPart = part[(threadIdx.x + threadIdx.y * blockDim.y) + (indexOffset * blockIdx.x)];

            valPart != 0 ? bg[id*3] = (bg[id] * 0.3) + (valPart * 0.7) + 0: 0;
            valPart != 0 ? bg[(id*3)+1] = (bg[id] * 0.3) + (valPart * 0.7) + 0: 0;
            valPart != 0 ? bg[(id*3)+2] = (bg[id] * 0.3) + (valPart * 0.7) + 0: 0;




            // bg[id] = blockDim.x-15;
            //bg[id] = threadIdx.x + (threadIdx.y * blockDim.y);
            //part[id] = blockDim.y;
      }
      """

    # def __init__(self):

    """Defining the basic CNN architecture of the Generator"""

    def __init__(self):
        print("Initializing ImageGenerator")

        """Defining the training scheme."""
        real_images = tf.placeholder(tf.float32, shape=[None, 784])
        z = tf.placeholder(tf.float32, shape=[None, 100])
        # print(z.type)
        print("z é")
        print(type(z))

        # Generator
        G = self.__innerSubGenerator(z)
        # print(G.type)
        print("G é")
        print(type(G))

        D_output_real, D_logits_real = self.__innerDiscriminator(real_images, None)
        D_output_fake, D_logits_fake = self.__innerDiscriminator(G, True)

        def loss_func(logits_in, labels_in):
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits_in, labels=labels_in))

        D_real_loss = loss_func(D_logits_real, tf.ones_like(D_logits_real) * (0.9))
        D_fake_loss = loss_func(D_logits_fake, tf.zeros_like(D_logits_real))
        D_loss = D_real_loss + D_fake_loss
        G_loss = loss_func(D_logits_fake, tf.ones_like(D_logits_fake))

        # Restoring the already trained model.
        tvars = tf.trainable_variables()
        d_vars = [var for var in tvars if 'dis' in var.name]
        g_vars = [var for var in tvars if 'gen' in var.name]
        print([v.name for v in d_vars])
        print([v.name for v in g_vars])
        # TF Session
        self.saver = tf.train.Saver(var_list=g_vars)


    def __innerSubGenerator(self, z, reuse=None):
        with tf.variable_scope('gen', reuse=reuse):
            hidden1 = tf.layers.dense(inputs=z, units=128)
            # Leaky Relu
            alpha = 0.01
            hidden1 = tf.maximum(alpha * hidden1, hidden1)
            hidden2 = tf.layers.dense(inputs=hidden1, units=128)

            hidden2 = tf.maximum(alpha * hidden2, hidden2)
            output = tf.layers.dense(hidden2, units=784, activation=tf.nn.tanh)
            return output

    def __innerDiscriminator(self, x, reuse=None):
        with tf.variable_scope('dis', reuse=reuse):
            hidden1 = tf.layers.dense(inputs=x, units=128)
            # Leaky Relu
            alpha = 0.01
            hidden1 = tf.maximum(alpha * hidden1, hidden1)

            hidden2 = tf.layers.dense(inputs=hidden1, units=128)
            hidden2 = tf.maximum(alpha * hidden2, hidden2)

            logits = tf.layers.dense(hidden2, units=1)
            output = tf.sigmoid(logits)

            return output, logits

    def discriminate(self, images):
        # image = np.array([1, 784], dtype=np.float64)
        images = np.asarray(images)
        print("Dados em Discriminate:")
        print(images.shape)
        print(images.dtype)
        print(images.size)
        # image_ph = tf.placeholder(tf.float32, shape=[None, 784])
        with tf.Session() as sess:
            self.saver.restore(sess, self.TRAINED_MODEL_DIR)

            # result, logits = sess.run(self.__innerDiscriminator(image_ph, reuse=True), feed_dict={image_ph: images})
            result, logits = sess.run(self.__innerDiscriminator(images, reuse=True))
        return result


    def generateSubImage(self, numImages, resolutionSubImage, imageType, reshape=True, convertUChar=True):

        # Selecting the resolution index, 28x28, 32x32 and 48x48.
        if (resolutionSubImage == 0):
            self.resolution = 28
        elif (resolutionSubImage == 1):
            self.resolution = 32
        elif (resolutionSubImage == 2):
            self.resolution = 48

        # Selecting the type of the primary image dataset.
        if (imageType == 0):
            dataset = "MNIST"
        elif (imageType == 1):
            dataset = "NIST"
        elif (imageType == 2):
            dataset = "CIFAR-10"

        # mnistDataSet = input_data.read_data_sets(self.MNIST_DIR, one_hot=True)




        new_samples = []

        # Creating the TF placeholder
        z = tf.placeholder(tf.float32, shape=[None, 100])
        ipt = tf.placeholder(tf.float32, shape=[None, 784])

        with tf.Session() as sess:
            self.saver.restore(sess, self.TRAINED_MODEL_DIR)

            for x in range(numImages):
                sample_z = np.random.uniform(-1, 1, size=(1, 100))

                gen_sample = sess.run(self.__innerSubGenerator(z, reuse=True), feed_dict={z: sample_z})
                # gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: imgList[x].reshape(1, 100)})
                # Test
                # list1 = [gen_sample]
                # list1 = np.asarray(gen_sample)
                # print("list1")
                # print(list1.shape)
                # init = tf.global_variables_initializer()
                # sess.run(init)
                # # Discriminating
                # result, logits = sess.run(self.__innerDiscriminator(ipt.initialized_value(), reuse=True)
                #                                                                 , feed_dict={ipt: list1})
                # print("Generated result")
                # print(result)

                if (reshape):
                    # new_samples.append(img_as_ubyte(gen_sample.reshape(28, 28)))
                    gen_sample = gen_sample.reshape(28, 28)
                if (convertUChar):
                    gen_sample = img_as_ubyte(gen_sample)

                new_samples.append(gen_sample)


            # result, logits = sess.run(self.__innerDiscriminator(new_samples, reuse=True), feed_dict={image_ph: images})
        return new_samples

    """Receiving a list of (sub)images, with each in the 28x28 shape and in uint8 format."""
    def generateFullImage(self, subImages, bgImage):
        numSubImages = len(subImages)

        # TODO Check if necessary to reshape and to convert the subImages.
        # Reshaping the subImages
        # for i in range(0, numSubImages):
        #     listaImagens.append(img_as_ubyte(mnist.train.images[i].reshape(28, 28)))

        """Preparing PyCUDA setup."""
        program1 = SourceModule(self.kernel1_code)

        # TODO Check if is really necessary convert to a numpy array.
        # Convert the python list to numpy array
        npArrayImages = np.asarray(subImages)

        tamLinha = np.array([1024], np.uint8)

        # Creating an array to offset the two cordinates in the background
        offsetArray = np.random.randint(800, size=(2, numSubImages))

        # Allocating GPU memory
        imgPart1_gpu = cuda.mem_alloc(npArrayImages.nbytes)
        bg_gpu = cuda.mem_alloc(bgImage.nbytes)
        tamLinha_gpu = cuda.mem_alloc(tamLinha.nbytes)
        offsetArray_gpu = cuda.mem_alloc(offsetArray.nbytes)

        cuda.memcpy_htod(bg_gpu, bgImage)
        cuda.memcpy_htod(imgPart1_gpu, npArrayImages)
        cuda.memcpy_htod(tamLinha_gpu, tamLinha)
        cuda.memcpy_htod(offsetArray_gpu, offsetArray)

        # TODO Calcular de acordo com a resolução do imgPart
        totalThreadSize = 28 * 28
        totalBlockSize = totalThreadSize
        gridSize = totalThreadSize / totalBlockSize
        # TODO Calcular de acordo com threadSize dinamicamente
        blockSpec = (28, 28, 1)
        gridSpec = (numSubImages, 1)

        # Invoking kernel
        kernelFunc = program1.get_function('printPart')
        kernelFunc(bg_gpu, imgPart1_gpu, offsetArray_gpu, tamLinha, block=blockSpec, grid=gridSpec)

        # Var to receive from GPU the altered image
        completeImage_host = np.empty_like(bgImage)

        cuda.memcpy_dtoh(completeImage_host, bg_gpu)

        print("Offsets:\n")
        print(offsetArray)

        return completeImage_host
    """Adiciona ruídos às imagens submetidas.
        - resize = float value from -0.5 to 0.5
        - bright = 
        - contrast = 
    """
    def addNoise(self, images, resize, rotate, brightness, contrast):
        # First image as model for shape
        rows, cols = images[0].shape

        numImages = len(images)

        if(resize != 0):
            print("\nInitial image shape: ")
            print(images[0].shape)

            oldSize = images[0][0].shape[0]
            # Calculating the new size
            if(resize > 0):
            #     First image as model for shape
                print("Aumento")
                newSize = int(oldSize * (resize+1))
            else:
                print("Diminui")
                newSize = oldSize - int(oldSize * (resize*-1))

            print("New size: "+str(newSize))

            newImgList = []
            for i in range(numImages):
                newImg = cv.resize(images[i], (newSize, newSize))

                # Crop to the old image size
                if(newSize > oldSize):
                    startLine = startCol = int((newSize - oldSize)/2)
                    endLine = endCol = startCol + oldSize
                    newImg = newImg[startLine:endLine, startCol:endCol]
                else:
                    startLine = startCol = int((oldSize - newSize)/2)
                    endLine = endCol = (oldSize - newSize - startCol) * -1
                    # print("\nParams: ")
                    # print(str(startLine) + ':' + str(endLine) + " - " + str(startCol) + ":" + str(endCol))
                    zeroMatrix = np.zeros((oldSize, oldSize))
                    zeroMatrix[startLine:endLine, startCol:endCol] = newImg
                    newImg = zeroMatrix

                images[i] = newImg

        if(rotate != 0):
            M = cv.getRotationMatrix2D((cols/2, rows/2), rotate, 1)
            i = 0
            for img in images:
                # img = cv.warpAffine(img, M, (cols, rows))
                newImg = cv.warpAffine(img, M, (cols, rows))
                images[i] = newImg
                i += 1

        if(brightness != 0 or contrast != 0):
            print("Brightness in")
            print(images[0].dtype)
            print(images[0][18])
            print(images[0][19])

            for i in range(len(images)):
                images[i] = cv.add(images[i], brightness)
                images[i] = cv.multiply(images[i], contrast)

            print("Post processing")
            print(images[0][18])
            print(images[0][19])

        return images