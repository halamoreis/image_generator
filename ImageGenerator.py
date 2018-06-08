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
import math

from skimage import img_as_ubyte
from skimage import img_as_float
from tensorflow.examples.tutorials.mnist import input_data


class ImageGenerator:
    # Default values
    MNIST_DIR = "./MNIST_data/"
    CNN_MODEL_DIR = 'models/cnn-model-5000.ckpt'
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

    simple_kernel_noise_code = """
    __global__ void addNoise(unsigned char *images, int *bright, float *contrast)
    {
            //Image size 28*28 = 784
            int indexOffset = 784;
            
            // Number of offsets
            //int offsetx = offsets[blockIdx.x];
            //int offsety = offsets[blockIdx.x + 4];
            
            //Rotating the destination cordinates
            
            
            //The index of the current pixel
            int pxIndex = (threadIdx.x + threadIdx.y * blockDim.y) + (indexOffset * blockIdx.x);
                        
            int pxVal = images[pxIndex];
            
            //Apply B & C
            pxVal = (int) (pxVal + bright[blockIdx.x]) * contrast[blockIdx.x];
            images[pxIndex] = pxVal > 255 ? 255 : pxVal;
    }
    """

    kernel_rotate_code = """
        __global__ void addNoise(unsigned char *images, unsigned char *outputImages, float *sin, float *cos
        , int *bright, float *contrast)
        {
                int hwidth = 14;
                int hheight = 14;
                int x = threadIdx.x;
                int y = threadIdx.y;
                
                int xt = x - hwidth;
                int yt = y - hheight;
                
                // float angle = rotateAngle[blockIdx.x];
                
                //double sinma = sin(-angle/2);
                //double cosma = cos(-angle/2);
                float sinma = sin[blockIdx.x];
                float cosma = cos[blockIdx.x];
                
                    
                
                int xs = (int)round((cosma * xt - sinma * yt) + hwidth);
                int ys = (int)round((sinma * xt + cosma * yt) + hheight);

                int indexOffset = 784;        
        
                if(xs >= 0 && xs < 28 && ys >= 0 && ys < 28) {
                    /* set target pixel (x,y) to color at (xs,ys) */
                    int pxIndex = (threadIdx.x + threadIdx.y * blockDim.y) + (indexOffset * blockIdx.x);
                    int newIndex = (xs + ys * blockDim.y) + (indexOffset * blockIdx.x);
                    
                    outputImages[newIndex] = images[pxIndex];
                } else {
                    /* set target pixel (x,y) to some default background */
                    int pxIndex = (threadIdx.x + threadIdx.y * blockDim.y) + (indexOffset * blockIdx.x);
                    /*
                    if(x == 0 && y ==0)
                        outputImages[pxIndex] = angle;
                    else*/
                        outputImages[pxIndex] = 0;
                }
                
        }
        """

    kernel_noise_code = """
      __global__ void addNoise(unsigned char *images, float *rotateAngle
      , int *bright, float *contrast)
      {
            //Image size 28*28 = 784
            int indexOffset = 784;
            
            // Number of offsets
            //int offsetx = offsets[blockIdx.x];
            //int offsety = offsets[blockIdx.x + 4];
            
            //Rotating the destination cordinates
            //width / 2 -> 28/2 = 14
            //int hwidth = 14;
            //int hheight = 14;
            
            //x = threadIdx.x && y = threadIdx.y
            int xt = threadIdx.x - 14;
            int yt = threadIdx.y - 14;
            
            float angle = rotateAngle[blockIdx.x];
            
            float sinma = sin(-angle);
            float cosma = cos(-angle);
            
            int xs = (int)round((cosma * xt - sinma * yt) + 14);
            int ys = (int)round((sinma * xt + cosma * yt) + 14);
            
            //ID of the current image
            //int bgId = (threadIdx.x + offsetx) + ((threadIdx.y+offsety) * 1024);
            
            //The index of the current pixel
            //int pxIndex = (threadIdx.x + threadIdx.y * blockDim.y) + (indexOffset * blockIdx.x);
            int pxIndex;
            //int img = images[(threadIdx.x + threadIdx.y * blockDim.y) + (indexOffset * blockIdx.x)];
            int pxVal = images[pxIndex];
            
            //Apply B & C
            int newVal = (pxVal * contrast[blockIdx.x]) + bright[blockIdx.x];
            int blankVal = 0;

            //images[pxIndex] = (unsigned char) (pxVal + bright[blockIdx.x]) * contrast[blockIdx.x];
            
            //Writing to...
            if(xs >= 0 && xs < 28 && ys >= 0 && ys < 28) {
                pxIndex = (xs + ys * blockDim.y) + (indexOffset * blockIdx.x);
                images[pxIndex] = newVal;
            } else {
                pxIndex = (threadIdx.x + threadIdx.y * blockDim.y) + (indexOffset * blockIdx.x);
                images[pxIndex] = blankVal;
            }
            
            //images[pxIndex] = newVal > 255 ? 255 : newVal;
            //images[pxIndex] += 100;

      }
      """
    # Test

    """ __global__ void addNoise(unsigned char *img, unsigned char *bc, long int *offsets, unsigned char *lineSize)
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
        # print("z é")
        # print(type(z))

        # Generator
        G = self.__innerSubGenerator(z)
        # print(G.type)
        # print("G é")
        # print(type(G))

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
        self.generatorSaver = tf.train.Saver(var_list=g_vars)
        # self.dGeneratorSaver = tf.train.Saver(var_list=d_vars)

        """ The CNN Discriminator """
        print("Initializing the CNN Discriminator")




    def cnnDiscriminator(self, input):
        tf.reset_default_graph()

        # self.mnist = input_data.read_data_sets(self.MNIST_DIR, one_hot=True)



        def init_weights(shape):
            init_random_dist = tf.truncated_normal(shape, stddev=0.1)
            return tf.Variable(init_random_dist)

        def init_bias(shape):
            init_bias_vals = tf.constant(0.1, shape=shape)
            return tf.Variable(init_bias_vals)

        def conv2d(x, W):
            return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

        def max_pool_2by2(x):
            return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                                  strides=[1, 2, 2, 1], padding='SAME')

        """Using the conv2d function, we'll return an actual convolutional layer here that uses an ReLu activation."""
        def convolutional_layer(input_x, shape):
            W = init_weights(shape)
            b = init_bias([shape[3]])
            return tf.nn.relu(conv2d(input_x, W) + b)

        """This is a normal fully connected layer"""
        def normal_full_layer(input_layer, size):
            input_size = int(input_layer.get_shape()[1])
            W = init_weights([input_size, size])
            b = init_bias([size])
            return tf.matmul(input_layer, W) + b

        ### Placeholders
        self.cnnX = tf.placeholder(tf.float32, shape=[None, 784])
        y_true = tf.placeholder(tf.float32, shape=[None, 10])

        ### Creating the Layers
        x_image = tf.reshape(self.cnnX, [-1, 28, 28, 1])

        convo_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[6, 6, 32, 64])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_2_flat = tf.reshape(convo_2_pooling, [-1, 7 * 7 * 64])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        # NOTE THE PLACEHOLDER HERE!
        self.hold_prob = tf.placeholder(tf.float32)
        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=self.hold_prob)

        self.y_pred = normal_full_layer(full_one_dropout, 10)

        ### Loss Function
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=self.y_pred))

        ### Optimizer
        optimizer = tf.train.AdamOptimizer(learning_rate=0.0001)
        train = optimizer.minimize(cross_entropy)

        ### Saver
        self.CNNSaver = tf.train.Saver()

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            sess.run(init)
            # Restoring the session
            self.CNNSaver.restore(sess, self.CNN_MODEL_DIR)
            result = sess.run(self.y_pred, feed_dict={self.cnnX: input, self.hold_prob: 1.0})
            max = sess.run(tf.argmax(result, 1))

        # Returnin the classification and the calculated score.
        return max, result


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
        # print("Dados em Discriminate:")
        # print(images.shape)
        # print(images.dtype)
        # print(images.size)
        image_ph = tf.placeholder(tf.float32, shape=[None, 784])
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.dGeneratorSaver.restore(sess, self.TRAINED_MODEL_DIR)

            sess.run(init)
            result, logits = sess.run(self.__innerDiscriminator(image_ph, reuse=tf.AUTO_REUSE), feed_dict={image_ph: images})
            # result, logits = sess.run(self.__innerDiscriminator(images, reuse=True))
        # print "\nResult is"
        # print type(result)
        # print result
        # print "\nLogits is"
        # print type(logits)
        # print logits

        return result


    """ Generate single images of the selected image type database.
    Args:
        numImages (int): The number of images which is wanted.
        resolutionSubImage (int): The resolution of each image in integer value from 0 to 3. 0 means 20x20, 1=28x28,
        2=32x32 and 3=48x48.
        imageType (int): The second parameter.
        reshape (bool): If the image wil.
        convertUChar (bool): If the dtype of the numpy array have to be converted to uint8. Otherwise the array will be
        float.

    Returns:
        bool: The return value. True for success, False otherwise.

    """
    def  generateSubImage(self, numImages, resolutionSubImage, imageType, reshape=True, convertUChar=True):

        # Selecting the resolution index, 20x20, 28x28, 32x32 and 48x48.
        if (resolutionSubImage == 0):
            self.resolution = 20
        elif (resolutionSubImage == 1):
            self.resolution = 28
        elif (resolutionSubImage == 2):
            self.resolution = 32
        elif (resolutionSubImage == 3):
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
        image_ph = tf.placeholder(tf.float32, shape=[None, 784])

        init = tf.global_variables_initializer()

        with tf.Session() as sess:
            self.generatorSaver.restore(sess, self.TRAINED_MODEL_DIR)

            for x in range(numImages):
                sample_z = np.random.uniform(-1, 1, size=(1, 100))

                gen_sample = sess.run(self.__innerSubGenerator(z, reuse=True), feed_dict={z: sample_z})


                # gen_sample = sess.run(generator(z, reuse=True), feed_dict={z: imgList[x].reshape(1, 100)})
                # Test
                # list1 = [gen_sample]
                # list1 = np.asarray(gen_sample)
                # print("list1")
                # print(list1.shape)
                # # Discriminating
                # result, logits = sess.run(self.__innerDiscriminator(ipt.initialized_value(), reuse=True)
                #                                                                 , feed_dict={ipt: list1})
                # print("Generated result")
                # print(result)

                # normalShape =

                if (reshape):
                    # new_samples.append(img_as_ubyte(gen_sample.reshape(28, 28)))
                    gen_sample = gen_sample.reshape(28, 28)
                if (convertUChar):
                    gen_sample = img_as_ubyte(gen_sample)

                new_samples.append(gen_sample)

            sess.run(init)
            # Getting the reliability
            gen_reliability, dis_logits = sess.run(self.__innerDiscriminator(image_ph, reuse=tf.AUTO_REUSE), feed_dict={image_ph: np.asarray(new_samples).reshape([-1, 784])})

            # result, logits = sess.run(self.__innerDiscriminator(new_samples, reuse=True), feed_dict={image_ph: images})
        return new_samples, gen_reliability

    """ Same as in the generateSubImage method but computing a score to the generated image and passing a boundary value
        comparing to the given score.
        
    """
    def generateSubImageWPrejudice(self, numImages, resolutionSubImage, imageType, reshape=True, convertUChar=True, prejudiceLimit = 20):
        # Boundary value for prejudice



        # Try to generate 50% more images than requested, to then select the high scored images.
        numReqImages = numImages
        numImages = int(numImages*1.5)

        # Selecting the resolution index, 20x20, 28x28, 32x32 and 48x48.
        if (resolutionSubImage == 0):
            self.resolution = 20
        elif (resolutionSubImage == 1):
            self.resolution = 28
        elif (resolutionSubImage == 2):
            self.resolution = 32
        elif (resolutionSubImage == 3):
            self.resolution = 48

        # Selecting the type of the primary image dataset.
        if (imageType == 0):
            dataset = "MNIST"
        elif (imageType == 1):
            dataset = "NIST"
        elif (imageType == 2):
            dataset = "CIFAR-10"

        new_samples = []

        # Creating the TF placeholder
        z = tf.placeholder(tf.float32, shape=[None, 100])
        # ipt = tf.placeholder(tf.float32, shape=[None, 784])
        # image_ph = tf.placeholder(tf.float32, shape=[None, 784])

        # init = tf.global_variables_initializer()

        with tf.Session() as sess:
            self.generatorSaver.restore(sess, self.TRAINED_MODEL_DIR)
            wastedImages = 0

            for x in range(numImages):
                sample_z = np.random.uniform(-1, 1, size=(1, 100))

                gen_sample = sess.run(self.__innerSubGenerator(z, reuse=True), feed_dict={z: sample_z})
                # if( reshape ):
                #     gen_sample = gen_sample.reshape(28, 28)
                # if (convertUChar):
                #     gen_sample = img_as_ubyte(gen_sample)
                # print("Shape gen_sample")
                # print(gen_sample.shape)
                new_samples.append(gen_sample.reshape(784))
            print("\n\n")

            # sess.run(init)
            # Getting the reliability
            # gen_reliability, dis_logits = sess.run(self.__innerDiscriminator(image_ph, reuse=tf.AUTO_REUSE),
            #                                        feed_dict={image_ph: np.asarray(new_samples).reshape([-1, 784])})

            # result, logits = sess.run(self.__innerDiscriminator(new_samples, reuse=True), feed_dict={image_ph: images})
        # return new_samples, gen_reliability

        """ Get the score for each generated image to calculate the prejudice."""
        classif, score = self.cnnDiscriminator(np.asarray(new_samples).reshape(numImages, 784))
        # print(score.shape)
        # print(len(score))
        # print(score)
        #
        # new_samples = new_samples.reshape(len(new_samples), 784)
        selectedSamples = []
        for i in range(numImages):
            if(np.amax(score[i]) > prejudiceLimit):
                selectedSamples.append(new_samples[i])
                # print("Appending...")
                # print(type(new_samples[i]))
                # print(new_samples[i].shape)
        #
        # objectsToRemove = []
        #
        # for i in range(numImages):
        #     if(np.amax(score[i]) < prejudiceLimit):
        #         objectsToRemove.append(new_samples[i])
        #         print("Appending...")
        #         print(type(new_samples[i]))
        #         print(new_samples[i].shape)
        #
        # # Removing the objects
        # for obj in objectsToRemove:
        #     print("Removing...")
        #     print(type(obj.all()))
        #     print(obj.shape)
        #     new_samples.remove(obj.all())

        selectedSamples = np.asarray(selectedSamples)
        if (reshape):
            selectedSamples = selectedSamples.reshape(len(selectedSamples), 784)

        return selectedSamples, score

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


    """Receiving a list of (sub)images, with each in the 28x28 shape and in uint8 format."""
    # def addNoiseGPU(self, subImages, bgImage):
    def addNoiseGPU(self, images, resize, rotate, brightness, contrast):
        numImages = len(images)

        # TODO Check if necessary to reshape and to convert the subImages.
        # Reshaping the subImages
        # for i in range(0, numSubImages):
        #     listaImagens.append(img_as_ubyte(mnist.train.images[i].reshape(28, 28)))

        """Preparing PyCUDA setup."""
        # cudaDeformationProgram = SourceModule(self.kernel_noise_code)
        cudaDeformationProgram = SourceModule(self.simple_kernel_noise_code)
        cudaRotationProgram = SourceModule(self.kernel_rotate_code)

        # TODO Check if is really necessary convert to a numpy array.
        # Convert the python list to numpy array
        # npArrayImages = np.asarray(subImages)

        # print rotate

        # Calculate the sine an cosine
        # sin = np.float32(np.apply_along_axis(math.sin, 0, rotate))
        sin = np.asarray(map(math.sin, rotate), dtype=np.float32)
        cos = np.asarray(map(math.cos, rotate), dtype=np.float32)
        # cos = np.float32(np.apply_along_axis(math.cos, 0, rotate))

        # print("Senos:")
        # print(sin)

        """ Preparing the parameters to send to the GPU """
        rotateAngle = np.zeros(numImages, np.float)

        # Test the type of the params

        # Allocating GPU memory
        images_gpu = cuda.mem_alloc(images.nbytes)
        outputImages_gpu = cuda.mem_alloc(images.nbytes)
        sin_gpu = cuda.mem_alloc(sin.nbytes)
        cos_gpu = cuda.mem_alloc(cos.nbytes)
        rotateAngle_gpu = cuda.mem_alloc(rotate.nbytes)
        brightness_gpu = cuda.mem_alloc(brightness.nbytes)
        contrast_gpu = cuda.mem_alloc(contrast.nbytes)

        # Copying the HOST values to the GPU memory
        cuda.memcpy_htod(images_gpu, images)
        cuda.memcpy_htod(outputImages_gpu, images)
        cuda.memcpy_htod(sin_gpu, sin)
        cuda.memcpy_htod(cos_gpu, cos)
        cuda.memcpy_htod(rotateAngle_gpu, rotate)
        cuda.memcpy_htod(brightness_gpu, brightness)
        cuda.memcpy_htod(contrast_gpu, contrast)

        # TODO Calcular de acordo com a resolução do imgPart
        totalThreadSize = 28 * 28
        totalBlockSize = totalThreadSize
        gridSize = totalThreadSize / totalBlockSize
        # TODO Calcular de acordo com threadSize dinamicamente
        blockSpec = (28, 28, 1)
        gridSpec = (numImages, 1)

        # Invoking kernel
        kernelDefFunc = cudaDeformationProgram.get_function('addNoise')
        kernelRotFunc = cudaRotationProgram.get_function('addNoise')
        kernelDefFunc(images_gpu, brightness_gpu, contrast_gpu, block=blockSpec, grid=gridSpec)
        kernelRotFunc(images_gpu, outputImages_gpu, sin_gpu, cos_gpu, brightness_gpu, contrast_gpu, block=blockSpec, grid=gridSpec)

        # Var to receive from GPU the altered image
        alteredImages = np.empty_like(images)

        # cuda.memcpy_dtoh(alteredImages, images_gpu)
        cuda.memcpy_dtoh(alteredImages, outputImages_gpu)

        # print("Offsets:\n")
        # print(offsetArray)

        return alteredImages


    """Adiciona ruídos às imagens submetidas.
        Expects an image in a 2 dimensional array
        - resize = float value from -0.5 to 0.5
        - bright = 
        - contrast = 
    """
    def addNoise(self, images, resize, rotate, brightness, contrast):
        # First image as model for shape
        rows, cols = images[0].shape

        numImages = len(images)

        if(resize != 0):
            # print("\nInitial image shape: ")
            # print(images[0].shape)

            oldSize = images[0][0].shape[0]
            # Calculating the new size
            if(resize > 0):
                #     First image as model for shape
                #     print("Aumento")
                newSize = int(oldSize * (resize+1))
            else:
                # print("Diminui")
                newSize = oldSize - int(oldSize * (resize*-1))

            # print("New size: "+str(newSize))


            newImgList = []
            for i in range(numImages):
                if (newSize == oldSize):
                    newImg = images[i]
                else:
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
            # print("Brightness in")
            # print(images[0].dtype)
            # print(images[0][18])
            # print(images[0][19])
            if(images[0].dtype == np.float):
                print("Float!")

            for i in range(len(images)):
                # TODO Gamb convertendo para uint8 e depois retornando para float
                imgTemp = img_as_ubyte(images[i])
                imgTemp = cv.add(imgTemp, brightness)
                images[i] = img_as_float(cv.multiply(imgTemp, contrast))

            # print("Post processing")
            # print(images[0][18])
            # print(images[0][19])

        return images


    """Add noise with variated parameter to images.
        Expects a list of images (in a 2 dimensional array of int) and the list of params
        (with the same length as the list of images). 
        - resize = float value from -0.5 to 0.5
        - rotate = 
        - bright = 
        - contrast = 
    """
    def addNoiseToLists(self, images, resizeParams, rotateParams, brightnessParams, contrastParams):
        # print("Resize: " + str(resizeParams[0]) + ", " + str(resizeParams[49]))
        # print("Rotate: " + str(rotateParams[0]) + ", " + str(rotateParams[49]))
        # print("Bright: " + str(brightnessParams[0]) + ", " + str(brightnessParams[49]))
        # print("Contrast: " + str(contrastParams[0]) + ", " + str(contrastParams[49]))
        # print("---------------------------------")
        # First image as model for shape
        rows, cols = images[0].shape

        numImages = len(images)
        print("NumImages "+str(numImages))

        for i in range(numImages):
            # print("Index: "+str(i))
            # print(images[i].dtype)

            """ RESIZING """
            if(resizeParams[i] != 0):
                # print("\nInitial image shape: ")
                # print(images[0].shape)

                oldSize = images[i][0].shape[0]
                # Calculating the new size
                if(resizeParams[i] > 0):
                    #     First image as model for shape
                    #     print("Aumento")
                    newSize = int(oldSize * (resizeParams[i]+1))
                else:
                    # print("Diminui")
                    newSize = oldSize - int(oldSize * (resizeParams[i]*-1))

                # print("New size: "+str(newSize))


                # newImgList = []
                # for i in range(numImages):

                if (newSize == oldSize):
                    newImg = images[i]
                else:
                    # newImg = img_as_ubyte(cv.resize(images[i], (newSize, newSize), newImg))
                    newImg = np.zeros((newSize, newSize), dtype=np.uint8)
                    cv.resize(images[i], dsize=(newSize, newSize), dst=newImg)

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
                        zeroMatrix = np.zeros((oldSize, oldSize), dtype=np.uint8)
                        zeroMatrix[startLine:endLine, startCol:endCol] = newImg
                        newImg = zeroMatrix

                images[i] = newImg
                # print("\n\nResized")
                # print(images[i].dtype)
                # print(images[i])

            """ --- ROTATING ---"""
            # Applying the rotate param
            if(rotateParams[i] != 0):
                M = cv.getRotationMatrix2D((cols/2, rows/2), rotateParams[i], 1)

                # img = cv.warpAffine(img, M, (cols, rows))
                newImg = cv.warpAffine(images[i], M, (cols, rows))
                images[i] = newImg
                # print(images[i].dtype)


            # Applying the brightness and contrast params
            if(brightnessParams[i] != 0 or contrastParams[i] != 0):
                # print("Brightness in")
                # print(images[0].dtype)
                # print(images[0][18])
                # print(images[0][19])
                # print(type(images[i]))
                if(images[i].dtype == np.float):
                    print("Float!")
                    print(images[i])
                    exit(0)

                # for i in range(len(images)):
                # print(images[i])
                # print(type(images[i]))
                # print(images[i].dtype)
                # imgTemp = img_as_ubyte(images[i])
                # imgTemp = cv.add(imgTemp, brightnessParams[i])
                imgTemp = cv.add(images[i], brightnessParams[i])
                # images[i] = img_as_float(cv.multiply(imgTemp, contrastParams[i]))
                images[i] = cv.multiply(imgTemp, np.float64(contrastParams[i]))

                # print("Post processing")
                # print(images[0][18])

        return images