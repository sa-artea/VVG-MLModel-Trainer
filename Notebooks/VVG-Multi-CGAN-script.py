# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-toolsai.jupyter added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\..\..\AppData\Local\Temp\5f7876e2-8542-4bed-8fda-de404b9653ef'))
	print(os.getcwd())
except:
	pass
# %% [markdown]
# useful links:
# 
# - Data Preparation for Variable Length Input Sequences, URL: https://machinelearningmastery.com/data-preparation-variable-length-input-sequences-sequence-prediction/
# - Masking and padding with Keras, URL: https://www.tensorflow.org/guide/keras/masking_and_padding
# - Step-by-step understanding LSTM Autoencoder layers, URL: https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352XX, 
# - Understanding input_shape parameter in LSTM with Keras, URL: https://stats.stackexchange.com/questions/274478/understanding-input-shape-parameter-in-lstm-with-keras
# - tf.convert_to_tensor, URL: https://www.tensorflow.org/api_docs/python/tf/convert_to_tensor
# - ValueError: Failed to convert a NumPy array to a Tensor (Unsupported object type int) in Python, URL: https://datascience.stackexchange.com/questions/82440/valueerror-failed-to-convert-a-numpy-array-to-a-tensor-unsupported-object-type
# - How to Identify and Diagnose GAN Failure Modes, URL: https://machinelearningmastery.com/practical-guide-to-gan-failure-modes/
# - How to Develop a GAN for Generating MNIST Handwritten Digits
# , URL: https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
# - How to Visualize a Deep Learning Neural Network Model in Keras
# , URL: https://machinelearningmastery.com/visualize-deep-learning-neural-network-model-keras/
# - How to Implement GAN Hacks in Keras to Train Stable Models
# , URL: https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
# - Tips for Training Stable Generative Adversarial Networks
# , URL: https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
# - How to Implement GAN Hacks in Keras to Train Stable Models
# , URL: https://machinelearningmastery.com/how-to-code-generative-adversarial-network-hacks/
# - How to Configure Image Data Augmentation in Keras, URL: https://machinelearningmastery.com/how-to-configure-image-data-augmentation-when-training-deep-learning-neural-networks/
# 

# %%
"""
* Copyright 2020, Maestria de Humanidades Digitales,
* Universidad de Los Andes
*
* Developed for the Msc graduation project in Digital Humanities
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

# ===============================
# native python libraries
# ===============================
import re
import random
import math
import json
import csv
import cv2
import datetime
import copy
import gc
from statistics import mean
from collections import OrderedDict
from collections import Counter
from collections import deque

# ===============================
# extension python libraries
# ===============================
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile
import numpy as np
import matplotlib.pyplot as plt

# natural language processing packages
import gensim
from gensim import models
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

# downloading nlkt data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# sample handling sklearn package
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import RandomOverSampler
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain

# # Keras + Tensorflow ML libraries
import tensorflow as tf
# from tensorflow.keras.layers

# preprocessing and processing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

# models
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model

# shapping layers
from tensorflow.keras.layers import Masking
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Concatenate

# basic layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import TimeDistributed

# data processing layers
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SpatialDropout1D

# recurrent and convolutional layers
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.layers import UpSampling2D
from tensorflow.keras.layers import ZeroPadding2D

# activarion function
from tensorflow.keras.layers import LeakyReLU

# optimization loss functions
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import SGD # OJO!
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.optimizers import Adam # OJO!
from tensorflow.keras.optimizers import Adadelta # OJO!
from tensorflow.keras.optimizers import Adagrad # OJO!

# image augmentation and processing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# ===============================
# developed python libraries
# ===============================

# %% [markdown]
# # FUNCTION DEFINITION

# %%
'''
A UDF to convert input data into 3-D
array as required for LSTM network.

taken from https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
'''
def temporalize(data, lookback):
    output_X = list()
    for i in range(len(data)-lookback-1):
        temp = list()
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            temp.append(data[[(i+j+1)], :])
        temp = np.array(temp, dtype="object")
        output_X.append(temp)
    output_X = np.array(output_X, dtype="object")
    return output_X


# %%
# function to read the image from file with cv2
def read_img(img_fpn):
    ans = cv2.imread(img_fpn, cv2.IMREAD_UNCHANGED)
    return ans


# %%
# fuction to scale the image and reduce cv2
def scale_img(img, scale_pct):

    width = int(img.shape[1]*scale_pct/100)
    height = int(img.shape[0]*scale_pct/100)
    dim = (width, height)
    # resize image
    ans = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return ans


# %%
# function to standarize image, has 2 types, from 0 to 1 and from -1 to 1
def std_img(img, minv, maxv, stype="std"):
    ans = None
    rangev = maxv - minv

    if stype == "std":
        ans = img.astype("float32")/float(rangev)
    
    elif stype == "ctr":
        rangev = float(rangev/2)
        ans = (img.astype("float32")-rangev)/rangev
    # ans = pd.Series(ans)
    return ans


# %%
# function to pad the image in the center
def pad_img(img, h, w, img_type):
    #  in case when you have odd number
    ans = None
    top_pad = np.floor((h - img.shape[0]) / 2).astype(np.uint8) # floor
    bottom_pad = np.ceil((h - img.shape[0]) / 2).astype(np.uint8)
    right_pad = np.ceil((w - img.shape[1]) / 2).astype(np.uint8)
    left_pad = np.floor((w - img.shape[1]) / 2).astype(np.uint8) # floor
    # print((top_pad, bottom_pad), (left_pad, right_pad))
    if img_type == "rgb":
        ans = np.copy(np.pad(img, ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)), mode="constant", constant_values=0.0))   
    if img_type == "bw":
        ans = np.copy(np.pad(img, ((int(top_pad), int(bottom_pad)), (int(left_pad), int(right_pad))), mode="constant", constant_values=0))

    return ans


# %%
def update_shape(src_df, img_col, shape_col):

    ans = src_df
    src_col = list(ans[img_col])
    tgt_col = list()

    # ansdict = {}
    for data in src_col:
        tshape = data.shape
        tgt_col.append(tshape)

    ans[shape_col] = tgt_col
    return ans


# %%
# function to padd the images in the dataset, needs the shape, the type of image and the src + tgt columns of the frame to work with
def padding_images(src_df, src_col, tgt_col, max_shape, img_type):
    # ans = src_df
    src_images = src_df[src_col]
    tgt_images = list()
    max_x, max_y = max_shape[0], max_shape[1]

    for timg in src_images:        
        pimg = pad_img(timg, max_y, max_x, img_type)
        tgt_images.append(pimg)

    src_df[tgt_col] = tgt_images
    return src_df


# %%
# function to load the images in in memory
def get_images(rootf, src_df, src_col, tgt_col, scale_pct):
    ans = src_df
    src_files = list(ans[src_col])
    tgt_files = list()

    # ansdict = {}
    for tfile in src_files:
        tfpn = os.path.join(rootf, tfile)
        timg = read_img(tfpn)
        timg = scale_img(timg, scale_pct)
        tgt_files.append(timg)

    ans[tgt_col] = tgt_files
    return ans


# %%
# function to standarize the images in the dataset, it has 2 options
def standarize_images(src_df, src_col, tgt_col, img_type, std_opt):
    src_images = src_df[src_col]
    tgt_images = list()

    for timg in src_images:
        # pcolor image
        if img_type == "rgb":
            timg = np.asarray(timg, dtype="object")
        
        # b&w image
        if img_type == "rb":
            timg = np.asarray(timg) #, dtype="uint8")
            timg = timg[:,:,np.newaxis]
            timg = np.asarray(timg, dtype="object")
        
        # std_opt affect the standarization results
        # result 0.0 < std_timg < 1.0
        # result -1.0 < std_timg < 1.0
        std_timg = std_img(timg, 0, 255, std_opt)
        tgt_images.append(std_timg)

    src_df[tgt_col] = tgt_images
    return src_df


# %%
# function than rotates the original image to create a new example
def syth_rgb_img(data):

    samples = expand_dims(data, 0)
    datagen = ImageDataGenerator(rotation_range=90)
    ans = datagen.flow(samples, batch_size=1)
    ans = ans[0].astype("uint8")
    ans = np.squeeze(ans, 0)
    return ans


# %%
# function to get the max shape in the image dataset
def get_mshape(shape_data, imgt):

    max_x, max_y, max_ch = 0, 0, 0
    shape_data = list(shape_data)
    ans = None

    if imgt == "rgb":

        for tshape in shape_data:
            # tshape = eval(tshape)
            tx, ty, tch = tshape[0], tshape[1], tshape[2]

            if tx > max_x:
                max_x = tx
            if ty > max_y:
                max_y = ty
            if tch > max_ch:
                max_ch = tch
            
        ans = (max_x, max_y, max_ch)
    
    elif imgt == "bw":

        for tshape in shape_data:
            # tshape = eval(tshape)
            tx, ty = tshape[0], tshape[1]

            if tx > max_x:
                max_x = tx
            if ty > max_y:
                max_y = ty
            
        ans = (max_x, max_y)
        
    return ans


# %%
'''
A UDF to convert input data into 3-D
array as required for LSTM network.

taken from https://towardsdatascience.com/step-by-step-understanding-lstm-autoencoder-layers-ffab055b6352
'''
def temporalize(data, lookback):
    output_X = list()
    for i in range(len(data)-lookback-1):
        temp = list()
        for j in range(1,lookback+1):
            # Gather past records upto the lookback period
            temp.append(data[[(i+j+1)], :])
        temp = np.array(temp, dtype="object")
        output_X.append(temp)
    output_X = np.array(output_X, dtype="object")
    return output_X


# %%
# format the pandas df data into usable word dense vector representation, YOU NEED IT FOR THE CSV to be useful!
def format_dvector(work_corpus):

    ans = list()
    for dvector in work_corpus:
        dvector = eval(dvector)
        dvector = np.asarray(dvector)
        ans.append(dvector)
    ans = np.asarray(ans, dtype="object")
    return ans


# %%
# funct to concatenate all label columns into one for a single y in ML training, returns a list
def concat_labels(row, cname):

    ans = list()
    for c in cname:
        r = row[c]
        r = eval(r)
        ans = ans + r

    return ans


# %%
# function to save the ML model
def save_model(model, m_path, m_file):

    fpn = os.path.join(m_path, m_file)
    fpn = fpn + ".h5"
    # print(fpn)
    model.save(fpn)


# %%
# function to load the ML model
def load_model(m_path, m_file):

    fpn = os.path.join(m_path, m_file)
    fpn = fpn + ".h5"
    model = keras.models.load_model(fpn)
    return model


# %%
# function to cast dataframe and avoid problems with keras
def cast_batch(data):

    cast_data = list()

    if len(data) >= 2:

        for d in data:
            d = np.asarray(d).astype("float32")
            cast_data.append(d)

    return cast_data


# %%
# function to select real elements to train the discriminator
def gen_real_samples(data, sample_size, half_batch):

    real_data = list()
    rand_index = np.random.randint(0, sample_size, size=half_batch)

    # need at leas X, y
    # posible combinations are:
    # X_img/X_txt, y
    # X_img/X_txt, X_labels, y
    # X_img, X_txt, X_labels, y
    if len(data) >= 2:
        # selectinc the columns in the dataset
        for d in data:
            # td_real = d[rand_index]
            td_real = copy.deepcopy(d[rand_index])
            real_data.append(td_real)

    # casting data
    real_data = cast_batch(real_data)

    return real_data


# %%
# function to create fake elements to train the discriminator
def gen_fake_samples(gen_model, dataset_shape, half_batch):

    # fake data
    fake_data = None
    # conditional labels for the gan model
    conditional = dataset_shape.get("conditioned")
    # configuratin keys for the generator
    latent_shape = dataset_shape.get("latent_shape")
    cat_shape = dataset_shape.get("cat_shape")
    label_shape = dataset_shape.get("label_shape")
    data_cols = dataset_shape.get("data_cols")

    # generator config according to the dataset
    # X:images -> y:Real/Fake
    if data_cols == 2:
        # random textual latent space 
        latent_text = gen_latent_txt(latent_shape, half_batch)
        # marking the images as fake in all accounts
        y_fake = gen_fake_negclass(cat_shape, half_batch)
        # random generated image from the model
        Xi_fake = gen_model.predict(latent_text)
        # fake samples
        fake_data = (Xi_fake, y_fake)

    # X_img, X_labels(classification), y (fake/real)
    elif (conditional == True) and data_cols == 3:
        # random textual latent space 
        latent_text = gen_latent_txt(latent_shape, half_batch)
        # marking the images as fake in all accounts
        y_fake = gen_fake_negclass(cat_shape, half_batch)
        # marking all the images with fake labels
        Xl_fake = gen_fake_labels(label_shape, half_batch)

        # random generated image from the model
        Xi_fake = gen_model.predict([latent_text, Xl_fake])
        # fake samples
        fake_data = (Xi_fake, Xl_fake, y_fake)

    elif (conditional == False) and data_cols == 3:
        
        # random textual latent space 
        latent_text = gen_latent_txt(latent_shape, half_batch)
        # marking the images as fake in all accounts
        y_fake = gen_fake_negclass(cat_shape, half_batch)
        # random generated image + text from the model
        Xi_fake, Xt_fake = gen_model.predict(latent_text)
        # fake samples
        fake_data = (Xi_fake, Xt_fake, y_fake)

    # X_img(rgb), X_txt(text), X_labels(classification), y (fake/real)
    elif data_cols == 4:

        # random textual latent space 
        latent_text = gen_latent_txt(latent_shape, half_batch)
        # marking the images as fake in all accounts
        y_fake = gen_fake_negclass(cat_shape, half_batch)
        # marking all the images with fake labels
        Xl_fake = gen_fake_labels(label_shape, half_batch)

        # random generated image from the model
        Xi_fake, Xt_fake = gen_model.predict([latent_text, Xl_fake])
        # fake samples 
        fake_data = (Xi_fake, Xt_fake, Xl_fake, y_fake)

    # casting data type
    fake_data = cast_batch(fake_data)
    
    return fake_data


# %%
# function to create inputs to updalte the GAN generator
def gen_latent_data(dataset_shape, batch_size):

    # latent data
    latent_data = None

    # conditional labels for the gan model
    conditional = dataset_shape.get("conditioned")
    # configuratin keys for the generator
    latent_shape = dataset_shape.get("latent_shape")
    cat_shape = dataset_shape.get("cat_shape")
    label_shape = dataset_shape.get("label_shape")
    data_cols = dataset_shape.get("data_cols")

    # generator config according to the dataset
    # X:images -> y:Real/Fake
    if data_cols == 2:
        # random textual latent space 
        latent_text = gen_latent_txt(latent_shape, batch_size)
        # marking the images as fake in all accounts
        y_gen = gen_fake_posclass(cat_shape, batch_size)
        # fake samples
        latent_data = (latent_text, y_gen)

    # X_img, X_labels(classification), y (fake/real)
    elif data_cols == 3 and (conditional == True):
        # random textual latent space 
        latent_text = gen_latent_txt(latent_shape, batch_size)
        # marking the images as fake in all accounts
        y_gen = gen_fake_posclass(cat_shape, batch_size)
        # marking all the images with fake labels
        Xl_gen = gen_fake_labels(label_shape, batch_size)
        # gen samples
        latent_data = (latent_text, Xl_gen, y_gen)

    elif data_cols == 3 and (conditional == False):
        # random textual latent space 
        latent_text = gen_latent_txt(latent_shape, batch_size)
        # marking the images as fake in all accounts
        y_gen = gen_fake_posclass(cat_shape, batch_size)
        # fake samples
        latent_data = (latent_text, y_gen)

    # X_img(rgb), X_txt(text), X_labels(classification), y (fake/real)
    elif data_cols == 4:
        # random textual latent space 
        latent_text = gen_latent_txt(latent_shape, batch_size)
        # marking the images as fake in all accounts
        y_gen = gen_fake_posclass(cat_shape, batch_size)
        # marking all the images with fake labels
        Xl_gen = gen_fake_labels(label_shape, batch_size)
        # gen samples
        latent_data = (latent_text, Xl_gen, y_gen)

    return latent_data
# latent_gen = gen_latent_txt(latent_shape, batch_size)
# create inverted category for the fake noisy text
# y_gen = get_fake_positive(cat_shape[0], batch_size)


# %%
# function to generate random/latent text for the GAN generator
def gen_latent_txt(latent_shape, n_samples):

    ans = None
    for i in range(n_samples):

        noise = np.random.normal(0.0, 1.0, size=latent_shape)
        if ans is None:
            txt = np.expand_dims(noise, axis=0)
            ans = txt
        else:
            img = np.expand_dims(txt, axis=0)
            ans = np.concatenate((ans, txt), axis=0)
    return ans


# %%
# tfunction to smooth the fake positives
def smooth_positives(y):
	return y - 0.3 + (np.random.random(y.shape)*0.5)


# %%
# function to smooth the fake negatives
def smooth_negatives(y):
	return y + np.random.random(y.shape)*0.3


# %%
# function to smooth the data labels
def smooth_labels(y):
    # label smoothing formula
    # alpha: 0.0 -> original distribution, 1.0 uniform distribution
    # K: number of label classes
    # y_ls = (1 - alpha) * y_hot + alpha / K
    alpha = 0.3
    K = y[0].shape[0]
    ans = (1-alpha)*y + alpha/K
    return ans


# %%
# generate fake true categories for the generator
def gen_fake_posclass(cat_shape, batch_size):

    sz = (batch_size, cat_shape[0])
    ans = np.ones(sz)
    # smoothing fakes
    ans = smooth_positives(ans)
    ans = ans.astype("float32")
    return ans


# %%
# generate fake negative category to train the GAN
def gen_fake_negclass(cat_shape, batch_size):

    sz = (batch_size, cat_shape[0])
    ans = np.zeros(sz)
    ans = smooth_negatives(ans)
    ans = ans.astype("float32")
    return ans


# %%
# function to generate fake labels to train the GAN
def gen_fake_labels(label_shape, batch_size):

    sz = (batch_size, label_shape[0])
    ans = np.random.randint(0,1, size=sz)
    ans = smooth_labels(ans)
    ans = ans.astype("float32")
    return ans


# %%
# function to create text similar to the original one with 5% of noise
def syth_text(data, nptc=0.05):

    ans = None
    noise = np.random.normal(0, nptc, data.shape)
    ans = data + noise
    return ans


# %%
# synthetizing a noisy std image from real data
def syth_std_img(data):

    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=10)
    # datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, vertical_flip=True)
    ans = datagen.flow(samples, batch_size=1)
    ans = ans[0].astype("float32")
    ans = np.squeeze(ans, 0)
    return ans


# %%
# function to create new categories with some noise, default 5%
def syth_categories(data, nptc=0.05):

    ans = None
    noise = np.random.normal(0, nptc, data.shape)
    ans = data + noise
    return ans


# %%
# function to artificially span a batch with some noise and alterations by an specific number
# TODO fix because this is an old version with no flexibility
def expand_samples(data, synth_batch):

    X_txt = data[0]
    X_img = data[1]
    y = data[2]
    labels = data[3]

    # creating the exapnded batch response
    Xe_txt, Xe_img, ye, lbe = None, None, None, None

    # iterating in the original batch
    for Xtt, Xit, yt, lb in zip(X_txt, X_img, y, labels):

        # temporal synth minibatch per original image
        synth_Xt, synth_Xi, synth_y, synth_lb = None, None, None, None

        # synthetizing artificial data for the batch
        for i in range(synth_batch):

            # generating first element
            if (synth_Xt is None) and (synth_Xi is None) and (synth_y is None) and (synth_lb is None):
                # gen text
                gen_Xt = copy.deepcopy(Xtt)
                gen_Xt = np.array(gen_Xt)
                gen_Xt = np.expand_dims(gen_Xt, axis=0)
                synth_Xt = gen_Xt

                # gen images
                gen_Xi = syth_std_img(Xit)
                gen_Xi = np.expand_dims(gen_Xi, axis=0)
                synth_Xi = gen_Xi

                # gen category
                gen_yt = syth_categories(yt)
                gen_yt = np.expand_dims(gen_yt, axis=0)
                synth_y = gen_yt

                # gen labels
                gen_lb = syth_categories(lb)
                gen_lb = np.expand_dims(gen_lb, axis=0)
                synth_lb = gen_lb

            # generatin the rest of the elements
            else:
                # gen text
                gen_Xt = syth_text(Xtt)
                gen_Xt = np.expand_dims(gen_Xt, axis=0)
                synth_Xt = np.concatenate((synth_Xt, gen_Xt), axis=0)

                # gen images
                gen_Xi = syth_std_img(Xit)
                gen_Xi = np.expand_dims(gen_Xi, axis=0)
                synth_Xi = np.concatenate((synth_Xi, gen_Xi), axis=0)

                # gen category
                gen_yt = syth_categories(yt)
                gen_yt = np.expand_dims(gen_yt, axis=0)
                synth_y = np.concatenate((synth_y, gen_yt), axis=0)
        
                # gen labels
                gen_lb = syth_categories(lb)
                gen_lb = np.expand_dims(gen_lb, axis=0)
                synth_lb = np.concatenate((synth_lb, gen_lb), axis=0)

        # adding the first part to the training batch
        if (Xe_txt is None) and (Xe_img is None) and (ye is None) and (lbe is None):
            # adding text
            Xe_txt = synth_Xt
            # adding images
            Xe_img = synth_Xi
            # adding categories
            ye = synth_y
            # adding labels
            lbe = synth_lb

        # adding the rest of the batch
        else:
            # adding text
            Xe_txt = np.concatenate((Xe_txt, synth_Xt), axis=0)
            # adding images
            Xe_img = np.concatenate((Xe_img, synth_Xi), axis=0)
            # adding category
            ye = np.concatenate((ye, synth_y), axis=0)
            # adding labels
            lbe = np.concatenate((lbe, synth_lb), axis=0)

    Xe_txt, Xe_img, ye, lbe = cast_batch(Xe_txt, Xe_img, ye, lbe)

    e_data = (Xe_txt, Xe_img, ye, lbe)

    return e_data


# %%
# def drift_labels(Xt_real, Xi_real, y_real, Xt_fake, Xi_fake, y_fake, batch_size, drift_pct):
def drift_labels(real_data, fake_data, batch_size, drift_pct):

    # setting the size for the drift labels
    drift_size = int(math.ceil(drift_pct*batch_size))
    # random index for drift elements!!!
    rand_drifters = np.random.choice(batch_size, size=drift_size, replace=False)
    # print("batch size", batch_size, "\nrandom choise to change", drift_size, "\n", rand_drifters)

    # if the dataset has at leas X, y... NEED TO PASS A GOOD ORDER
    if (len(real_data) and len(fake_data)) >= 2:

        # iterating over the random choose index
        for drift in rand_drifters:

            # taking one real + fake column at a time
            # X_img/txt, y
            # X_img/txt, X_labels, y
            # X_img, X_txt, X_labels, y
            for real_col, fake_col in zip(real_data, fake_data):

                # copying real data in temporal var
                temp_drift = copy.deepcopy(real_col[drift])
                # replacing real with fakes
                real_col[drift] = copy.deepcopy(fake_col[drift])
                # updating fakes with temporal original
                fake_col[drift] = temp_drift

    return real_data, fake_data


# %%
# function to standarize image, has 2 types, from 0 to 1 and from -1 to 1
def inv_std_img(img, minv, maxv, stype="std"):
    ans = None
    rangev = maxv - minv

    if stype == "std":
        ans = img*rangev
        ans = np.asarray(ans).astype("uint8")

    elif stype == "ctr":
        rangev = float(rangev/2)
        ans = img+rangev
        ans = ans*rangev
        ans = np.asarray(ans).astype("uint8")

    return ans


# %%
# the function takes the ideas array, shape and configuration to render them into human understandable lenguage
# it select n number of ideas and plot them, for images, for text and for both
def plot_ideas(ideas, ideas_shape, train_cfg, test_cfg):

    # get the index of random ideas in the set
    n_sample = test_cfg.get("n_samples")
    ideas_size = ideas.shape[0]
    # choosing non repeated ideas in the set
    rand_ideas = np.random.choice(ideas_size, size=n_samples*n_samples, replace=False)

    # if the ideas are images or text
    if len(ideas) == 1:
        
        current_shape = ideas[0].shape

        if current_shape == ideas_shape.get("txt_shape"):
            fig = render_wordcloud(ideas, rand_ideas, test_cfg)
            save_ideas(fig, train_cfg, test_cfg)

        elif current_shape == ideas_shape.get("img_shape"):
            fig = render_painting(ideas, rand_ideas, test_cfg)
            save_ideas(fig, train_cfg, test_cfg)

    # if the ideas are images + text
    elif len(ideas) == 2:
        fig_txt = render_wordcloud(ideas, rand_ideas, test_cfg)
        fig_img = render_painting(ideas, rand_ideas, test_cfg)
        save_ideas((fig_img, fig_txt), train_cfg, test_cfg)


# %%
# this function takes the selected ideas and transform them into pytlot objects
def render_painting(ideas, rand_index, test_cfg):

    # get important data for iterating
    n_sample = test_cfg.get("n_samples")
    report_fn_path = test_cfg.get("report_fn_path")
    epoch = test_cfg.get("epoch")
    example_size = examples.shape[0]

    # prep the figure
    fig, ax = plt.subplots(n_sample,n_sample, figsize=(20,20))
    fig.patch.set_facecolor("xkcd:white")

    # plot images
    for i in range(n_sample*n_sample):
        # define subplot
        plt.subplot(n_sample, n_sample, 1+i)

        # getting the images from sample
        rand_i = rand_index[i]
        gimg = ideas[rand_i]
        gimg = inv_std_img(gimg, 0, 255, "ctr")

        # turn off axis
        plt.axis("off")
        plt.imshow(gimg) #, interpolation="nearest")

    # plot leyend
    fig.suptitle("GENERATED PAINTINGS", fontsize=50)
    fig.legend()

    # save plot to file
    plot_name = "GAN-Gen-img-epoch%03d" % int(epoch)
    plot_name = plot_name + ".png"
    fpn = os.path.join(report_fp_name, plot_name)
    plt.savefig(fpn)
    plt.close()

    return fig


# %%
# this function takes the selected ideas and translate them into pytplot objects
def render_wordcloud(ideas, rand_index, test_cfg):
    # get important data for iterating
    n_sample = test_cfg.get("n_samples")
    example_size = examples.shape[0]

    # prep the figure
    fig, ax = plt.subplots(n_sample,n_sample, figsize=(20,20))
    fig.patch.set_facecolor("xkcd:white")

    # plot images
    for i in range(n_sample*n_sample):
        # define subplot
        plt.subplot(n_sample, n_sample, 1+i)

        # getting the images from sample
        rand_i = rand_index[i]
        gimg = ideas[rand_i]
        gimg = inv_std_img(gimg, 0, 255, "ctr")

        # turn off axis
        plt.axis("off")
        plt.imshow(gimg) #, interpolation="nearest")

    # plot leyend
    fig.suptitle("GENERATED TEXT", fontsize=50)
    fig.legend()

    return fig


# %%
# this function takes the pyplot objects and saves them into a file
def save_idea():
    pass


# %%
# this function loads the model known lexicon into the a dictionary for the world cloud to translate
def load_lexicon():
    pass


# %%
# this function takes the idtf dense word vector representacion and translate it to human lenguage using the kown lexicon
def translate_from_lexicon():
    pass


# %%
# function to plot the generated images within a training epoch
def plot_gen_images(examples, epoch, report_fp_name, n_sample):

    # get important data for iterating
    example_size = examples.shape[0]
    og_shape = examples[0].shape
    rand_img = np.random.choice(example_size, size=n_sample*n_sample, replace=False) 
    # (0, example_size, size=n_sample*n_sample)

    # prep the figure
    fig, ax = plt.subplots(n_sample,n_sample, figsize=(20,20))
    fig.patch.set_facecolor("xkcd:white")

    # plot images
    for i in range(n_sample*n_sample):
        # define subplot
        plt.subplot(n_sample, n_sample, 1+i)

        # getting the images from sample
        rand_i = rand_img[i]
        gimg = examples[rand_i]
        gimg = inv_std_img(gimg, 0, 255, "ctr")

        # turn off axis
        plt.axis("off")
        plt.imshow(gimg) #, interpolation="nearest")

    # plot leyend
    fig.suptitle("GENERATED PAINTINGS", fontsize=50)
    fig.legend()

    # save plot to file
    plot_name = "GAN-Gen-img-epoch%03d" % int(epoch)
    plot_name = plot_name + ".png"
    fpn = os.path.join(report_fp_name, plot_name)
    plt.savefig(fpn)
    plt.close()


# %%
# create a line plot of loss for the gan and save to file
def plot_metrics(disr_hist, disf_hist, gan_hist, report_fp_name, epoch):

    # reporting results
    disr_hist = np.array(disr_hist)
    disf_hist = np.array(disf_hist)
    gan_hist = np.array(gan_hist)

    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
    fig.patch.set_facecolor("xkcd:white")

    # loss
    ax1.plot(disr_hist[:,1], "royalblue", label="Loss: R-Dis")
    ax1.plot(disf_hist[:,1], "crimson", label="Loss: F-Dis")
    ax1.plot(gan_hist[:,1], "blueviolet", label="Loss: GAN/Gen")
    # ax1.plot(gan_hist[:], "blueviolet", label="Loss: GAN/Gen")

    # acc_
    ax2.plot(disr_hist[:,0], "royalblue", label="Acc: R-Dis")
    ax2.plot(disf_hist[:,0], "crimson", label="Acc: F-Dis")
    ax2.plot(gan_hist[:,0], "blueviolet", label="Acc: GAN/Gen")

    # plot leyend
    fig.suptitle("LEARNING BEHAVIOR", fontsize=20)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title("Loss")
    ax2.set_title("Accuracy")
    ax1.set(xlabel = "Epoch [cycle]", ylabel = "Loss")
    ax2.set(xlabel = "Epoch [cycle]", ylabel = "Acc")
    fig.legend()

    # save plot to file
    plot_name = "GAN-learn-curve-epoch%03d" % int(epoch)
    plot_name = plot_name + ".png"
    fpn = os.path.join(report_fp_name, plot_name)
    plt.savefig(fpn)
    plt.close()


# %%
# function to calculate the loss and accuracy avg in multiple batchs of an epoch
def epoch_avg(log):
    loss, acc = None, None

    # if acc and loss are present to avg
    if type(log[0]) is list:
        if len(log) > 0:

            acc_list = list()
            loss_list = list()

            for l in log:
                ta = l[0]
                tl = l[1]

                acc_list.append(ta)
                loss_list.append(tl)

            loss, acc = mean(loss_list), mean(acc_list)
        return loss, acc
    
    else:
        # if only loss is present
        if len(log) > 0:

            loss_list = list()

            for l in log:
                loss_list.append(l)

            loss = mean(loss_list)
        return loss


# %%
# function to save model, needs the dirpath, the name and the datetime to save
def export_model(model, models_fp_name, filename, datetime):

    ss = True
    sln = True
    fext = "png"
    fpn = filename + "-" + datetime
    fpn = filename + "." + fext
    fpn = os.path.join(models_fp_name, fpn)
    plot_model(model, to_file=fpn, show_shapes=ss, show_layer_names=sln)


# %%
# function to format data to save in file
def format_metrics(disr_history, disf_history, gan_history):

    headers, data = None, None

    disr_hist = np.array(disr_history)
    disf_hist = np.array(disf_history)
    gan_hist = np.array(gan_history)

    # formating file headers
    headers = ["dis_loss_real", "dis_acc_real", "dis_loss_fake", "dis_acc_fake", "gen_gan_loss", "gen_gan_acc"]
    # headers = ["dis_loss_real", "dis_acc_real", "dis_loss_fake", "dis_acc_fake", "gen_gan_loss",] # "gen_gan_acc"]

    # formating fake discriminator train data
    drhl = disr_hist[:,1]
    drha = disr_hist[:,0]

    # formating real discrimintator train data
    dfhl = disf_hist[:,1]
    dfha = disf_hist[:,0]

    # formating gan/gen train data
    # gghl = gan_hist[:]# .flatten()
    gghl = gan_hist[:,1]
    ggha = gan_hist[:,0]

    # adding all formatted data into list
    data = np.column_stack((drhl, drha, dfhl, dfha, gghl, ggha))
    # data = np.column_stack((drhl, drha, dfhl, dfha, gghl)) #, ggha))

    return data, headers


# %%
# function to write data in csv file
def write_metrics(data, headers, report_fn_path, filename):

    # print(report_fn_path, filename)
    fpn = filename + "-train-history.csv"
    fpn = os.path.join(report_fn_path, fpn)

    history_df = pd.DataFrame(data, columns=headers)
    tdata = history_df.to_csv(
                            fpn,
                            sep=",",
                            index=False,
                            encoding="utf-8",
                            mode="w",
                            quoting=csv.QUOTE_ALL
                            )


# %%
# function to safe the loss/acc logs in training for the gan/gen/dis models
def save_metrics(disr_history, disf_history, gan_history, report_fn_path, filename):

    data, headers = format_metrics(disr_history, disf_history, gan_history)
    write_metrics(data, headers, report_fn_path, filename)


# %%
# function to know the time between epochs or batchs it return the new time for a new calculation
def lapse_time(last_time, epoch):

    now_time = datetime.datetime.now()
    deltatime = now_time - last_time
    deltatime = deltatime.total_seconds()
    deltatime = "%.2f" % deltatime
    msg = "Epoch:%3d " % int(epoch+1)
    msg = msg + "elapsed time: " + str(deltatime) + " [s]"
    print(msg)
    return now_time


# %%
# function to test the model while training
def test_model(gen_model, dis_model, data, data_shape, test_cfg): #batch_size, synth_batch, report_fn_path): #train_cfg)

    dataset_size = test_cfg.get("dataset_size")
    batch_size = test_cfg.get("batch_size")
    synth_batch = test_cfg.get("synth_batch")
    epoch = int(test_cfg.get("current_epoch"))
    report_fn_path = test_cfg.get("report_fn_path")
    gen_samples = test_cfg.get("gen_sample_size") 
    balance_batch = test_cfg.get("balance_batch")

    # select real txt2img for discrimintator
    real_data = gen_real_samples(data, dataset_size, batch_size)
    # create false txt for txt2img for generator
    fake_data = gen_fake_samples(gen_model, data_shape, batch_size)

    # expand the training sample for the discriminator
    if synth_batch > 1:
        real_data = expand_samples(real_data, synth_batch)
        fake_data = expand_samples(fake_data, synth_batch)

    # balance training samples for the discriminator
    if balance_batch == True:
        real_data = balance_samples(real_data)
        fake_data = balance_samples(fake_data)
        
    # print(Xt_real.shape, Xi_real.shape, y_real.shape, yl_real.shape)
    # print(Xt_fake.shape, Xi_fake.shape, y_fake.shape, yl_fake.shape)

    # gen data
    X_test = fake_data[0]
    split_batch = int(batch_size/2)

    # plotting gen images
    # plot_gen_ideas(fake_data, epoch, report_fn_path, gen_samples)
    plot_gen_images(X_test, epoch, report_fn_path, gen_samples)

    test_real, test_fake = None, None

    if len(data) == 2:
        test_real, test_fake = test_gan(dis_model, real_data, fake_data, batch_size)

    elif len(data) == 3 and data_shape.get("conditioned") == True:
        test_real, test_fake = test_cgan(dis_model, real_data, fake_data, batch_size)

    elif len(data) == 3 and data_shape.get("conditioned") == False:
        test_real, test_fake = test_multi_gan(dis_model, real_data, fake_data, batch_size)

    elif len(data) == 4:
        test_real, test_fake = test_multi_cgan(dis_model, real_data, fake_data, batch_size)

    # summarize discriminator performance
    print("Batch Size %d -> Samples: Fake: %d & Real: %d" % (batch_size*synth_batch, split_batch, split_batch))
    print(">>> Test Fake -> Acc: %.3f || Loss: %.3f" % (test_fake[1], test_fake[0]))
    print(">>> Test Real -> Acc: %.3f || Loss: %.3f" % (test_real[1], test_real[0]))
    # print(">>> Test Gen -> Acc: %.3f || Loss: %.3f" % (test_cgen[1], test_cgen[0]))


# %%
# special function to train the GAN
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
# def train(gen_model, dis_model, gan_model, X_img, X_txt, y, labels, epochs, batch_size, save_intervas, fn_config):
def training_model(gen_model, dis_model, gan_model, data, train_cfg): # epochs, batch_size, save_intervas, fn_config

    # sample size
    dataset_size = train_cfg.get("dataset_size")

    # data shape for the generator
    data_shape = {
        "latent_shape": train_cfg.get("latent_shape"),
        "cat_shape": train_cfg.get("cat_shape"),
        "txt_shape": train_cfg.get("txt_shape"),
        "label_shape": train_cfg.get("label_shape"),
        "conditioned": train_cfg.get("conditioned"),
        "data_cols": train_cfg.get("data_cols"),
        }

    # augmentation factor
    synth_batch = train_cfg.get("synth_batch")
    balance_batch = train_cfg.get("balance_batch")
    n = train_cfg.get("gen_sample_size")

    epochs = train_cfg.get("epochs")
    batch_size = train_cfg.get("batch_size")
    half_batch = int(batch_size/2)
    batch_per_epoch = int(dataset_size/batch_size)
    # fake/real batch division
    real_batch = int((batch_size*synth_batch)/2)

    # train config
    model_fn_path = train_cfg.get("models_fn_path")
    report_fn_path = train_cfg.get("report_fn_path")
    dis_model_name = train_cfg.get("dis_model_name")
    gen_model_name = train_cfg.get("gen_model_name")
    gan_model_name = train_cfg.get("gan_model_name")
    check_intervas = train_cfg.get("check_epochs")
    save_intervas = train_cfg.get("save_epochs")
    max_models = train_cfg.get("max_models")
    pretrain = train_cfg.get("pretrained")

	# prepare lists for storing stats each epoch
    disf_hist, disr_hist, gan_hist = list(), list(), list()
    train_time = None

    # train dict config
    test_cfg = {
        "report_fn_path": report_fn_path,
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "synth_batch": synth_batch,
        "gen_sample_size": train_cfg.get("gen_sample_size"),
        "epoch": None,
    }

    # iterating in training epochs:
    for ep in range(epochs+1):
        # epoch logs
        ep_disf_hist, ep_disr_hist, ep_gan_hist = list(), list(), list()
        train_time = datetime.datetime.now()

        # iterating over training batchs
        for batch in range(batch_per_epoch):

            # select real txt2img for discrimintator
            real_data = gen_real_samples(data, dataset_size, half_batch)
            # create false txt for txt2img for generator
            fake_data = gen_fake_samples(gen_model, data_shape, half_batch)

            # expand the training sample for the discriminator
            if synth_batch > 1:
                real_data = expand_samples(real_data, synth_batch)
                fake_data = expand_samples(fake_data, synth_batch)

            # balance training samples for the discriminator
            if balance_batch == True:
                real_data = balance_samples(real_data)
                fake_data = balance_samples(fake_data)

            # print(Xt_real.shape, Xi_real.shape, y_real.shape, yl_real.shape)
            # print(Xt_fake.shape, Xi_fake.shape, y_fake.shape, yl_fake.shape)
            # print(real_data[0].shape, fake_data[0].shape)
            # print(real_data[1].shape, fake_data[1].shape)
            # drift labels to confuse the model
            real_data, fake_data = drift_labels(real_data, fake_data, half_batch, 0.05)

            # TODO transfor this in 1 function train_model()...
            dhf, dhr, gh = None, None, None

            if len(data) == 2:
                dhf, dhr, gh = train_gan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 3 and data_shape.get("conditioned") == True:
                dhf, dhr, gh = train_cgan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 3 and data_shape.get("conditioned") == False:
                dhf, dhr, gh = train_multi_gan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 4:
                dhf, dhr, gh = train_multi_cgan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)
            
            print("training metrics!!!! [dis-fake, dis-real, GAN/gen]")
            print(dhf, dhr, gh)

            # epoch log
            ep_disr_hist.append(dhf)
            ep_disf_hist.append(dhr)
            ep_gan_hist.append(gh)

			# print('>%d, %d/%d, dis_=%.3f, gen=%.3f' % (ep+1, batch+1, bat_per_epo, dis_history, gen_history))
            log_msg = ">>> Epoch: %d, B/Ep: %d/%d, Batch S: %d" % (ep+1, batch+1, batch_per_epoch, batch_size*synth_batch)
            log_msg = "%s -> [R-Dis loss: %.3f, acc: %.3f]" % (log_msg, dhr[0], dhr[1])
            log_msg = "%s || [F-Dis loss: %.3f, acc: %.3f]" % (log_msg, dhf[0], dhf[1])
            log_msg = "%s || [Gen loss: %.3f, acc: %.3f]" % (log_msg, gh[0], gh[1])
            print(log_msg)

        # record history for epoch
        disr_hist.append(epoch_avg(ep_disr_hist))
        disf_hist.append(epoch_avg(ep_disf_hist))
        gan_hist.append(epoch_avg(ep_gan_hist))

		# evaluate the model performance sometimes
        if (ep) % check_intervas == 0:
            print("Epoch:", ep+1, "Saving the training progress...")
            test_cfg["epoch"] = ep
            # test_model(gen_model, dis_model, data, data_shape, test_cfg) #, synth_batch)
            test_model(gen_model, dis_model, data, data_shape, train_cfg, test_cfg)
            # plot_metrics(disr_hist, disf_hist, gan_hist, report_fn_path, ep)
            # save_metrics(disr_hist, disf_hist, gan_hist, report_fn_path, gan_model_name)

		# saving the model sometimes
        if (ep) % save_intervas == 0:
            epoch_sufix = "-epoch%d" % int(ep)
            # epoch_sufix = "-last"
            epoch_sufix = str(epoch_sufix)
            dis_mn = dis_model_name + epoch_sufix
            gen_mn = gen_model_name + epoch_sufix
            gan_mn = gan_model_name + epoch_sufix

            dis_path = os.path.join(model_fn_path, "Dis")
            gen_path = os.path.join(model_fn_path, "Gen")
            gan_path = os.path.join(model_fn_path, "GAN")

            save_model(dis_model, dis_path, dis_mn)
            save_model(gen_model, gen_path, gen_mn)
            save_model(gan_model, gan_path, gan_mn)
        
        train_time = lapse_time(train_time, ep)


# %%
def train_gan(dis_model, gan_model, real_data, fake_data, batch_size, dataset_shape):

    # real data asignation
    Xi_real = real_data[0]
    y_real = real_data[1]

    # fake data asignation
    Xi_fake = fake_data[0]
    y_fake = fake_data[1]

    # train for real samples batch
    dhr = dis_model.train_on_batch(Xi_real, y_real)
    # train for fake samples batch
    dhf = dis_model.train_on_batch(Xi_fake, y_fake)

    # prepare text and inverted categories from the latent space as input for the generator
    latent_gen, y_gen = gen_latent_data(dataset_shape, batch_size)

    # update the generator via the discriminator's error
    gh = gan_model.train_on_batch(latent_gen, y_gen)

    return dhf, dhr, gh


# %%
def train_cgan(dis_model, gan_model, real_data, fake_data, batch_size, dataset_shape):

    # real data asignation
    Xi_real = real_data[0]
    yl_real = real_data[1]
    y_real = real_data[2]

    # fake data asignation
    Xi_fake = fake_data[0]
    yl_fake = fake_data[1]
    y_fake = fake_data[2]

    # train for real samples batch
    dhr = dis_model.train_on_batch([Xi_real, yl_real], y_real)
    # train for fake samples batch
    dhf = dis_model.train_on_batch([Xi_fake, yl_fake], y_fake)

    # prepare text and inverted categories from the latent space as input for the generator
    latent_gen, yl_gen, y_gen = gen_latent_data(dataset_shape, batch_size)

    # update the generator via the discriminator's error
    gh = gan_model.train_on_batch([latent_gen, yl_gen], y_gen)

    return dhf, dhr, gh


# %%
def train_multi_cgan(dis_model, gan_model, real_data, fake_data, batch_size, dataset_shape):

    # real data asignation
    Xi_real = real_data[0]
    Xt_real = real_data[1]
    Xl_real = real_data[2]
    y_real = real_data[3]

    # fake data asignation
    Xi_fake = fake_data[0]
    Xt_fake = fake_data[1]
    Xl_fake = fake_data[2]
    y_fake = fake_data[3]

    # train for real samples batch
    dhr = dis_model.train_on_batch([Xi_real, Xt_real, Xl_real], y_real)
    # train for fake samples batch
    dhf = dis_model.train_on_batch([Xi_fake, Xt_fake, Xl_fake], y_fake)

    # prepare text and inverted categories from the latent space as input for the generator
    latent_gen, yl_gen, y_gen = gen_latent_data(dataset_shape, batch_size)

    # update the generator via the discriminator's error
    gh = gan_model.train_on_batch([latent_gen, yl_gen], y_gen)

    return dhf, dhr, gh


# %%
def test_gan(dis_model, real_data, fake_data, batch_size):
    
    # drift labels to confuse the model
    real_data, fake_data = drift_labels(real_data, fake_data, batch_size, 0.05)

    # real data asignation
    Xi_real = real_data[0]
    y_real = real_data[1]

    # fake data asignation
    Xi_fake = fake_data[0]
    y_fake = fake_data[1]

    # evaluate model
    test_real = dis_model.evaluate(Xi_real, y_real, verbose=0)
    test_fake = dis_model.evaluate(Xi_fake, y_fake, verbose=0)

    return test_real, test_fake


# %%
def test_cgan(dis_model, real_data, fake_data, batch_size):
    
    # drift labels to confuse the model
    real_data, fake_data = drift_labels(real_data, fake_data, batch_size, 0.05)

    # real data asignation
    Xi_real = real_data[0]
    Xl_real = real_data[1]
    y_real = real_data[2]

    # fake data asignation
    Xi_fake = fake_data[0]
    Xl_fake = fake_data[1]
    y_fake = fake_data[2]

    # evaluate model
    test_real = dis_model.evaluate([Xi_real, Xl_real], y_real, verbose=0)
    test_fake = dis_model.evaluate([Xi_fake, Xl_fake], y_fake, verbose=0)

    return test_real, test_fake


# %%
def test_multi_cgan(dis_model, real_data, fake_data, batch_size):

    # drift labels to confuse the model
    real_data, fake_data = drift_labels(real_data, fake_data, batch_size, 0.05)

    # real data asignation
    Xi_real = real_data[0]
    Xt_real = real_data[1]
    Xl_real = real_data[2]
    y_real = real_data[3]

    # fake data asignation
    Xi_fake = fake_data[0]
    Xt_fake = fake_data[1]
    Xl_fake = fake_data[2]
    y_fake = fake_data[3]

    # evaluate model
    test_real = dis_model.evaluate([Xi_real, Xt_real, Xl_real], y_real, verbose=0)
    test_fake = dis_model.evaluate([Xi_fake, Xt_fake, Xl_fake], y_fake, verbose=0)

    return test_real, test_fake

# %% [markdown]
# # EXEC SCRIPT
# 
# ## Dataset prep

# %%
# variable definitions
# root folder
dataf = "Data"

# subfolder with predictions txt data
imagef = "Img"

# report subfolder
reportf = "Reports"

#  subfolder with the CSV files containing the ML pandas dataframe
trainf = "Train"
testf = "Test"

# subfolder for model IO
modelf = "Models"

# dataframe file extension
fext = "csv"

imgf = "jpg"

rgb_sufix = "rgb"
bw_sufix = "bw"

# standard sufix
stdprefix = "std-"

# ml model useful data
mltprefix = "ml-"

# report names
# timestamp = datetime.date.today().strftime("%d-%b-%Y")
timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

sample_sufix = "Small"
# sample_sufix = "Large"
# sample_sufix = "Paintings"
imgf_sufix = "Img-Data-"
text_sufix = "Text-Data-"

# std-VVG-Gallery-Text-Data-Paintings
gallery_prefix = "VVG-Gallery-"

# dataframe file name
text_fn = stdprefix + gallery_prefix + text_sufix + sample_sufix + "." + fext
imgf_fn = stdprefix + gallery_prefix + imgf_sufix + sample_sufix + "." + fext
valt_fn = "Validation-GAN-" + text_sufix + sample_sufix + "." + fext

# model names
dis_model_name = "VVG-Text2Img-CDiscriminator"
gen_model_name = "VVG-Text2Img-CGenerator"
gan_model_name = "VVG-Text2Img-CGAN"

# to continue training after stoping script
continue_training = True

# ramdom seed
randseed = 42

# sample distribution train vs test sample size
train_split = 0.80
test_split = 1.0 - train_split

# regex to know that column Im interested in
keeper_regex = r"(^ID$)|(^std_)"

imgt = rgb_sufix
# imgt = bw_sufix

# woring values for code
work_txtf, work_imgf, work_sufix, work_imgt = text_fn, imgf_fn, sample_sufix, imgt

print("=== working files ===")
print("\n", work_txtf, "\n", work_imgf, "\n", work_sufix, "\n", work_imgt, "\n", valt_fn)


# %%
root_folder = os.getcwd()
root_folder = os.path.split(root_folder)[0]
root_folder = os.path.normpath(root_folder)
print(root_folder)


# %%
# variable reading
# dataframe filepath for texttual data
text_fn_path = os.path.join(root_folder, dataf, trainf, work_txtf)
print(text_fn_path, os.path.exists(text_fn_path))

# dataframe filepath for img data
img_fn_path = os.path.join(root_folder, dataf, trainf, work_imgf)
print(img_fn_path, os.path.exists(img_fn_path))

# dataframe filepath form GAN data
val_fn_path = os.path.join(root_folder, dataf, testf, valt_fn)
print(val_fn_path, os.path.exists(val_fn_path))

# filepath for the models
model_fn_path = os.path.join(root_folder, dataf, modelf)
print(model_fn_path, os.path.exists(model_fn_path))

# filepath for the reports
report_fn_path = os.path.join(root_folder, dataf, reportf)
print(report_fn_path, os.path.exists(report_fn_path))


# %%
# rading training data
# loading textual file
text_df = pd.read_csv(
                text_fn_path,
                sep=",",
                encoding="utf-8",
                engine="python",
            )
text_cols = text_df.columns.values

# loading image file
img_df = pd.read_csv(
                img_fn_path,
                sep=",",
                encoding="utf-8",
                engine="python",
            )
img_cols = img_df.columns.values


# %%
idx_cols = list()

for tcol in text_cols:
    if tcol in img_cols:
        idx_cols.append(tcol)
print(idx_cols)

source_df = pd.merge(text_df, img_df, how="inner", on=idx_cols)


# %%
# checking everything is allrigth
img_df = None
text_df = None
source_df.info()


# %%
source_df = source_df.set_index("ID")


# %%
# reading images from folder and loading images into df
# working variables
src_col = work_imgt + "_img"
tgt_col = work_imgt + "_img" + "_data"
work_shape = work_imgt + "_shape"
scale = 50
print(src_col, tgt_col)
source_df = get_images(root_folder, source_df, src_col, tgt_col, scale)


# %%
# update image shape
source_df = update_shape(source_df, tgt_col, work_shape)


# %%
# searching the biggest shape in the image files
print(work_shape)
shape_data = source_df[work_shape]
max_shape = get_mshape(shape_data, work_imgt)
print(max_shape)


# %%
# padding training data according to max shape of the images in gallery
pad_prefix = "pad_"
conv_prefix = "cnn_"
src_col = work_imgt + "_img" + "_data"
tgt_col = pad_prefix + conv_prefix + src_col

print(src_col, tgt_col, work_imgt)
source_df = padding_images(source_df, src_col, tgt_col, max_shape, work_imgt)


# %%
# reading images from folder and stadarizing images into df
# working variables
print("standarizing regular images...")
src_col = work_imgt + "_img" + "_data"
tgt_col = "std_" + src_col

# source_df = standarize_images(source_df, src_col, tgt_col)


# %%
print("standarizing padded images...")
src_col = pad_prefix + conv_prefix + work_imgt + "_img" + "_data"
tgt_col = "std_" + src_col
print(src_col, tgt_col)

# std_opt = "std"
std_opt = "ctr"
source_df = standarize_images(source_df, src_col, tgt_col, work_imgt, std_opt)


# %%
# shuffle the DataFrame rows
source_df.info()


# %%
# cleaning memory
gc.collect()


# %%
# function to find a name of column names according to a regex
def get_keeper_cols(col_names, search_regex):
    ans = [i for i in col_names if re.search(search_regex, i)]
    return ans


# %%
# function to find the disperse columns in the df
def get_disperse_categories(src_df, keep_cols, max_dis, check_cols, ignore_col):

    ans = list()

    max_dis = 2
    tcount = 0

    while tcount < max_dis:
        for label_col in keep_columns:

            if label_col != ignore_col:

                label_count = src_df[label_col].value_counts(normalize=False)

                if tcount < label_count.shape[0] and (check_cols in label_col):
                    tcount = label_count.shape[0]
                    ans.append(label_col)
                # print("count values of", label_col, ":=", label_count.shape)#.__dict__)
        tcount = tcount + 1
    
    return ans


# %%
# function to remove the disperse columns from the interesting ones
def remove_disperse_categories(keep_columns, too_disperse):
    for too in too_disperse:
        keep_columns.remove(too)
    return keep_columns


# %%
def padding_corpus(train_df, dvector_col, pad_prefix):
    # getting the corpus dense vectors
    work_corpus = np.asarray(train_df[dvector_col], dtype="object")

    # converting list of list to array of array
    print("Original txt shape", work_corpus.shape)

    # padding the representation
    work_corpus = pad_sequences(work_corpus, dtype='object', padding="post")
    # print("Padded txt shape", work_corpus.shape)

    # creating the new column and saving padded data
    padded_col_dvector = pad_prefix + dvector_col

    # print(padded_col)
    train_df[padded_col_dvector] = list(work_corpus)
    print("Padded txt shape", work_corpus.shape)
    return train_df


# %%
def heat_categories(train_df, cat_cols, tgt_col):

    labels_data = train_df[cat_cols]
    labels_concat = list()

    # concatenating all category labels from dataframe
    for index, row in labels_data.iterrows():
        row = concat_labels(row, labels_cols)
        labels_concat.append(row)

    # print(len(labels_concat[0]), type(labels_concat[0]))
    # updating dataframe
    tcat_label_col = "std_cat_labels"
    train_df[tgt_col] = labels_concat

    return train_df


# %%
# function to adjust the textual data for the LSTM layers in the model
def format_corpus(corpus, timesteps, features):

    # preparation for reshape lstm model
    corpus = temporalize(corpus, timesteps)
    print(corpus.shape)

    corpus = corpus.reshape((corpus.shape[0], timesteps, features))
    print(corpus.shape)

    return corpus


# %%
# selecting data to train
# want to keep the columns starting with STD_
keep_columns = list(source_df.columns)
print("------ original input/interested columns ------")
print(keep_columns)

# create the columns Im interesting in
keep_columns = get_keeper_cols(keep_columns, keeper_regex)
# keep_columns = [i for i in df_columns if re.search(keeper_regex, i)]

print("\n\n------ Interesting columns ------")
print(keep_columns)


# %%
too_disperse = get_disperse_categories(source_df, keep_columns, 2, "std_cat_", "std_pad_cnn_rgb_img_data")
print(too_disperse)


# %%
# creating the training dataframe
keep_columns = remove_disperse_categories(keep_columns, too_disperse)
# keep_columns.remove("ID")
print("------ Interesting columns ------")
print(keep_columns)


# %%
# creating the training dataframe
train_df = pd.DataFrame(source_df, columns=keep_columns)


# %%
# shuffling the stuff
train_df = train_df.sample(frac = 1)
source_df = None
df_columns = list(train_df.columns)


# %%
train_df.info()


# %%
# getting the column with the relevant data to train
pad_regex = u"^std_pad_"
padimg_col = get_keeper_cols(df_columns, pad_regex)
padimg_col = padimg_col[0]
print("Padded image column in dataframe: ", str(padimg_col))


# %%
# getting the column with the relevant data to train
dvec_regex = u"^std_dvec"
dvector_col = get_keeper_cols(df_columns, dvec_regex)
dvector_col = dvector_col[0]
print("Dense vector column in dataframe: ", str(dvector_col))


# %%
# fix column data type
work_corpus = train_df[dvector_col]
work_corpus = format_dvector(work_corpus)


# %%
# changing type in dataframe
train_df[dvector_col] = work_corpus
work_corpus = None


# %%
# padding training data according to max length of text corpus
pad_prefix = "pad_"
recurrent_prefix = "lstm_"

train_df = padding_corpus(train_df, dvector_col, pad_prefix)


# %%
regular_img_col = "std_" + work_imgt + "_img" + "_data"
padded_img_col = "std_" + pad_prefix + conv_prefix + work_imgt + "_img" + "_data"
padded_col_dvector = pad_prefix + dvector_col


# %%
# getting the columns with the relevant labels to predict
print(keep_columns)
cat_regex = u"^std_cat_"
labels_cols = get_keeper_cols(keep_columns, cat_regex)
print("Classifier trainable labels in dataframe: ", str(labels_cols))

# updating dataframe with hot/concatenated categories
tcat_label_col = "std_cat_labels"
print("categories heat column:", tcat_label_col)
train_df = heat_categories(train_df, labels_cols, tcat_label_col)


# %%
# getting the columns with the relevant labels to predict
print(keep_columns)
labels_cols = [i for i in keep_columns if re.search(u"^std_cat_", i)]
print("Trainable labels columns in dataframe: ", str(labels_cols))

labels_data = train_df[labels_cols]
labels_concat = list()

# concatenating all category labels from dataframe
for index, row in labels_data.iterrows():
    row = concat_labels(row, labels_cols)
    labels_concat.append(row)


# %%
text_lstm_col = padded_col_dvector
print(text_lstm_col)


# %%
working_img_col = padded_img_col
# working_img_col = regular_img_col
print(working_img_col)


# %%
train_df.info()


# %%
gc.collect()


# %%
# creating Train/Test sample
# getting the X, y to train, as is autoencoder both are the same
og_shape = train_df[working_img_col][0].shape# y[0].shape
X_img_len = train_df[working_img_col].shape[0] #y.shape[0]
print(X_img_len, og_shape)

X_img = None

for img in train_df[working_img_col]:

    if X_img is None:
        img = np.expand_dims(img, axis=0)
        X_img = img
    else:
        img = np.expand_dims(img, axis=0)
        X_img = np.concatenate((X_img, img), axis=0)

print("final X_img shape", X_img.shape)
# y.shape = (1899, 800, 800, 3)


# %%
print(type(X_img[0]))
print(type(X_img[0][0]))
print(X_img[1].shape)


# %%
if len(X_img.shape) == 3:
    X_img = X_img[:,:,:,np.newaxis]


# %%
# y = train_df[working_img_col]
# y = np.expand_dims(y, axis=0)
y_labels = np.asarray([np.asarray(j, dtype="object") for j in train_df[tcat_label_col]], dtype="object")
print("y shape", y_labels.shape)


# %%
y = np.ones((y_labels.shape[0],1)).astype("float32")
print("y shape", y.shape)


# %%
print("y classification category")
print(type(y[0]))
print(type(y[0][0]))
print(y[1].shape)

print("y labels category")
print(type(y_labels[0]))
print(type(y_labels[0][0]))
print(y_labels[1].shape)


# %%
# creating Train/Test sample
# getting the X, y to train, as is autoencoder both are the same
X_txt = np.asarray([np.asarray(i, dtype="object") for i in train_df[text_lstm_col]], dtype="object")
# X = np.array(train_df[text_lstm_col]).astype("object")
# X = train_df[text_lstm_col]
print("final X_lstm shape", X_txt.shape)


# %%
print(type(X_txt[0]))
print(type(X_txt[0][0]))
print(X_txt[1].shape)


# %%
# timestep is the memory of what i read, this is the longest sentence I can remember in the short term
# neet to look for the best option, in small the max is 15
timesteps = 15

# features is the max length in the corpus, after padding!!!!
features = X_txt[0].shape[0]
print(timesteps, features)


# %%
X_txt = format_corpus(X_txt, timesteps, features)


# %%
print(X_txt.shape)


# %%
diff_txt = y.shape[0] - X_txt.shape[0]
print(diff_txt)


# %%
Xa = X_txt[-diff_txt:]
X_txt = np.append(X_txt, Xa, axis=0)
print(X_txt.shape)
Xa = None


# %%
print(X_txt.shape)
print(X_img.shape)
print(y.shape)
print(y_labels.shape)


# %%
print(X_txt[0].shape)
print(X_img[0].shape)
print(y[0].shape)
print(y_labels[0].shape)
txt_og_shape = X_txt[0].shape
img_og_shape = X_img[0].shape
cat_og_shape = y[0].shape
lab_og_shape = y_labels[0].shape


# %%
# Xt = X_txt # np.array(X).astype("object")
# Xi = X_img
# yt = y # np.array(y).astype("object")
# # ya = y[0:timesteps]
# train_df = None


# %%
gc.collect()

# %% [markdown]
# # ML Model Definition
# 
# ## Image GAN

# %%
# convolutional generator for images
def create_img_generator(latent_shape, model_cfg):

    # MODEL CONFIG
    # def of the latent space size for the input
    # input layer config, latent txt space
    latent_n = model_cfg.get("latent_img_size")
    in_lyr_act = model_cfg.get("input_lyr_activation")
    # latent img shape
    latent_img_shape = model_cfg.get("latent_img_shape")

    # hidden layer config
    filters = model_cfg.get("filters")
    ksize = model_cfg.get("kernel_size")
    stsize = model_cfg.get("stride")
    pad = model_cfg.get("padding")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    hid_ldrop = model_cfg.get("gen_dropout_rate")
    mval = model_cfg.get("mask_value")
    rs = model_cfg.get("return_sequences")
    lstm_units = model_cfg.get("lstm_neurons")

    # output layer condig
    out_filters = model_cfg.get("output_filters")
    out_ksize = model_cfg.get("output_kernel_size")
    out_stsize = model_cfg.get("output_stride")
    out_pad = model_cfg.get("output_padding")
    img_shape = model_cfg.get("output_shape")
    out_lyr_act = model_cfg.get("output_lyr_activation")
    # LAYER CREATION
    # input layer
    in_latent = Input(shape=latent_shape, name="ImgGenIn")

    # masking input text
    lyr1 = Masking(mask_value=mval, input_shape=latent_shape, 
                    name = "ImgGenMask_1")(in_latent) # concat1

    # intermediate recurrent layer
    lyr2 = LSTM(lstm_units, activation=in_lyr_act, 
                    input_shape=latent_shape, 
                    return_sequences=rs, name="ImgGenLSTM_2")(lyr1)

    # flatten from 2D to 1D
    lyr3 = Flatten(name="ImgGenFlat_3")(lyr2)

    # dense layer
    lyr4 = Dense(latent_n, 
                activation=hid_lyr_act, 
                name="ImgGenDense_4")(lyr3)
    
    # reshape layer 1D-> 2D (rbg image)
    lyr5 = Reshape(latent_img_shape, name="ImgGenReshape_5")(lyr4)

    # transpose conv2D layer
    lyr6 = Conv2DTranspose(int(filters/8), kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgGenConv2D_6")(lyr5)

    # batch normalization + drop layers to avoid overfit
    lyr7 = BatchNormalization(name="ImgGenBN_7")(lyr6)
    lyr8 = Dropout(hid_ldrop, name="ImgGenDrop_8")(lyr7)


    # transpose conv2D layer
    lyr9 = Conv2DTranspose(int(filters/4), kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgGenConv2D_9")(lyr8)

    # transpose conv2D layer
    lyr10 = Conv2DTranspose(int(filters/2), kernel_size=ksize, 
                            strides=out_stsize, activation=out_lyr_act, 
                            padding=out_pad, name="ImgGenConv2D_10")(lyr9)

    # batch normalization + drop layers to avoid overfit
    lyr11 = BatchNormalization(name="ImgGenBN_11")(lyr10)
    lyr12 = Dropout(hid_ldrop, name="ImgGenDrop_12")(lyr11)

    # transpose conv2D layer
    lyr13 = Conv2DTranspose(filters, kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgGenConv2D_13")(lyr12)

    # output layer
    out_img = Conv2D(out_filters, kernel_size=out_ksize, 
                        strides=out_stsize, activation=out_lyr_act, 
                        padding=out_pad, input_shape=img_shape, 
                        name="ImgGenOut")(lyr13)

    # MODEL DEFINITION
    model = Model(inputs=in_latent, outputs=out_img)
    return model


# %%
# convolutional discriminator for images
def create_img_discriminator(img_shape, model_cfg):

    # MODEL CONFIG
    # input layer config, image classification
    in_lyr_act = model_cfg.get("input_lyr_activation")
    in_filters = model_cfg.get("input_filters")
    in_ksize = model_cfg.get("input_kernel_size")
    in_stsize = model_cfg.get("input_stride")
    in_pad = model_cfg.get("input_padding")

    # hidden layer config
    filters = model_cfg.get("filters")
    ksize = model_cfg.get("kernel_size")
    stsize = model_cfg.get("stride")
    pad = model_cfg.get("padding")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    hid_ldrop = model_cfg.get("dis_dropout_rate")
    # mid neuron size
    mid_disn = model_cfg.get("mid_dis_neurons")
    hid_cls_act = model_cfg.get("dense_cls_activation")

    # output layer condig
    out_nsize = model_cfg.get("output_dis_neurons")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # LAYER CREATION
    # input layer
    in_img = Input(shape=img_shape, name="DisImgIn")

    # DISCRIMINATOR LAYERS
    # intermediate conv layer
    lyr1 = Conv2D(in_filters, kernel_size=in_ksize, 
                    padding=in_pad, activation=in_lyr_act, 
                    strides=in_stsize, name="ImgDisConv2D_1")(in_img)

    # intermediate conv layer
    lyr2 = Conv2D(int(filters/2), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgDisConv2D_2")(lyr1)

    # batch normalization + drop layers to avoid overfit
    lyr3 = BatchNormalization(name="ImgDisBN_3")(lyr2)
    lyr4 = Dropout(hid_ldrop, name="ImgDisDrop_4")(lyr3)

    # intermediate conv layer
    lyr5 = Conv2D(int(filters/4), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgDisConv2D_4")(lyr4)

    # intermediate conv layer
    lyr6 = Conv2D(int(filters/8), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgDisConv2D_5")(lyr5)

    # batch normalization + drop layers to avoid overfit
    lyr7 = BatchNormalization(name="ImgDisBN_6")(lyr6)
    lyr8 = Dropout(hid_ldrop, name="ImgDisDrop_7")(lyr7)

    # flatten from 2D to 1D
    lyr9 = Flatten(name="ImgDisFlat_8")(lyr8)

    # dense classifier layers
    lyr10 = Dense(int(mid_disn), activation=hid_cls_act, name="ImgDisDense_9")(lyr9)
    lyr11 = Dense(int(mid_disn/2), activation=hid_cls_act, name="ImgDisDense_10")(lyr10)
    # drop layer
    lyr12 = Dropout(hid_ldrop, name="ImgDisDrop_11")(lyr11)

    # dense classifier layers
    lyr13 = Dense(int(mid_disn/4), activation=hid_cls_act, name="ImgDisDense_12")(lyr12)
    lyr14 = Dense(int(mid_disn/8), activation=hid_cls_act, name="ImgDisDense_13")(lyr13)
    # drop layer
    lyr15 = Dropout(hid_ldrop, name="ImgDisDrop_14")(lyr14)

    # dense classifier layers
    lyr16 = Dense(int(mid_disn/16), activation=hid_cls_act, name="ImgDisDense_15")(lyr15)
    lyr17 = Dense(int(mid_disn/32), activation=hid_cls_act, name="ImgDisDense_16")(lyr16)

    # output layer
    out_cls = Dense(out_nsize, activation=out_lyr_act, name="ImgDisOut")(lyr17)

    # MODEL DEFINITION
    model = Model(inputs=in_img, outputs=out_cls)
    return model


# %%
def create_img_gan(gen_model, dis_model, gan_cfg):

    # getting GAN Config
    ls = gan_cfg.get("loss")
    opt = gan_cfg.get("optimizer")
    met = gan_cfg.get("metrics")

	# make weights in the discriminator not trainable
    dis_model.trainable = False
	# get noise and label inputs from generator model
    gen_noise = gen_model.input
    # get image output from the generator model
    gen_output = gen_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = dis_model(gen_output)
    # define gan model as taking noise and label and outputting a classification
    model = Model(gen_noise, gan_output)
    # compile model
    model.compile(loss=ls, optimizer=opt, metrics=met)
    # model.compile(loss=ls, optimizer=opt)
    return model

# %% [markdown]
# ## Text GAN

# %%
# LSTM generator for text
def create_txt_generator(latent_shape, model_cfg):

    # MODEL CONFIG
    # def of the latent space size for the input
    # input layer config, latent txt space
    mval = model_cfg.get("mask_value")
    in_rs = model_cfg.get("input_return_sequences")
    in_lstm = model_cfg.get("input_lstm_neurons")
    in_lyr_act = model_cfg.get("input_lyr_activation")

    # hidden layer config
    latent_n = model_cfg.get("mid_gen_neurons")
    latent_reshape = model_cfg.get("latent_lstm_reshape")
    lstm_units = model_cfg.get("lstm_neurons")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    hid_ldrop = model_cfg.get("gen_dropout_rate")
    mem_shape = model_cfg.get("memory_shape")
    rs = model_cfg.get("hidden_return_sequences")

    # output layer condig
    txt_shape = model_cfg.get("output_neurons")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # LAYER CREATION
    # input layer
    in_latent = Input(shape=latent_shape, name="TxtGenIn")

    # masking input text
    lyr1 = Masking(mask_value=mval, input_shape=latent_shape, 
                    name = "TxtGenMask_1")(in_latent) # concat1

    # intermediate recurrent layer
    lyr2 = LSTM(in_lstm, activation=in_lyr_act, 
                    input_shape=latent_shape, 
                    return_sequences=in_rs, 
                    name="TxtGenLSTM_2")(lyr1)

    # batch normalization + drop layers to avoid overfit
    lyr3 = BatchNormalization(name="TxtGenBN_3")(lyr2)
    lyr4 = Dropout(hid_ldrop, name="TxtGenDrop_4")(lyr3)

    # flatten from 2D to 1D
    lyr5 = Flatten(name="TxtGenFlat_5")(lyr4)

    # dense layer
    lyr6 = Dense(latent_n, 
                activation=hid_lyr_act, 
                name="TxtGenDense_6")(lyr5)

    # reshape layer 1D-> 2D (rbg image)
    lyr7 = Reshape(latent_reshape, name="TxtGenReshape_7")(lyr6)

    # batch normalization + drop layers to avoid overfit
    lyr8 = BatchNormalization(name="TxtGenBN_8")(lyr7)
    lyr9 = Dropout(hid_ldrop, name="TxtGenDrop_9")(lyr8)

    # intermediate recurrent layer
    lyr10 = LSTM(int(lstm_units/4), activation=hid_lyr_act, 
                    input_shape=mem_shape, 
                    return_sequences=rs, 
                    name="TxtGenLSTM_10")(lyr9)

    # intermediate recurrent layer
    lyr11 = LSTM(int(lstm_units/2), activation=hid_lyr_act, 
                    input_shape=mem_shape, 
                    return_sequences=rs, 
                    name="TxtGenLSTM_11")(lyr10)

    # batch normalization + drop layers to avoid overfit
    lyr12 = BatchNormalization(name="TxtGenBN_12")(lyr11)
    lyr13 = Dropout(hid_ldrop, name="TxtGenDrop_13")(lyr12)

    # output layer, dense time sequential layer.
    lyr14 = LSTM(lstm_units, activation=hid_lyr_act, 
                    input_shape=mem_shape, 
                    return_sequences=rs, 
                    name="TxtGenDrop_14")(lyr13)

    out_txt = TimeDistributed(Dense(txt_shape, activation=out_lyr_act), name = "GenTxtOut")(lyr14)

    # model definition
    model = Model(inputs=in_latent, outputs=out_txt)

    return model


# %%
# LSTM discriminator for text
def create_txt_discriminator(txt_shape, model_cfg):

    # MODEL CONFIG
    # def of the latent space size for the input
    # input layer config, latent txt space
    mval = model_cfg.get("mask_value")
    in_rs = model_cfg.get("input_return_sequences")
    in_lstm = model_cfg.get("input_lstm_neurons")
    in_lyr_act = model_cfg.get("input_lyr_activation")

    # hidden layer config
    lstm_units = model_cfg.get("lstm_neurons")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    hid_ldrop = model_cfg.get("dis_dropout_rate")
    mem_shape = model_cfg.get("memory_shape")
    rs = model_cfg.get("hidden_return_sequences")

    # mid neuron size
    mid_disn = model_cfg.get("mid_dis_neurons")
    hid_cls_act = model_cfg.get("dense_cls_activation")

    # output layer condig
    out_nsize = model_cfg.get("output_dis_neurons")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # LAYER CREATION
    # input layer
    in_txt = Input(shape=txt_shape, name="DisTxtIn")

    # DISCRIMINATOR LAYERS
    # masking input text
    lyr1 = Masking(mask_value=mval, input_shape=txt_shape, 
                    name = "TxtDisMask_1")(in_txt) # concat1

    # input LSTM layer
    lyr2 = LSTM(in_lstm, activation=in_lyr_act, 
                    input_shape=txt_shape, 
                    return_sequences=in_rs, 
                    name="TxtDisLSTM_2")(lyr1)

    # batch normalization + drop layers to avoid overfit
    lyr3 = BatchNormalization(name="TxtDisBN_3")(lyr2)
    lyr4 = Dropout(hid_ldrop, name="TxtDisDrop_4")(lyr3)

    # intermediate LSTM layer
    lyr5 = LSTM(int(lstm_units/2), 
                activation=hid_lyr_act, 
                input_shape=mem_shape, 
                return_sequences=rs, 
                name="TxtDisLSTM_5")(lyr4)

    # intermediate LSTM layer
    lyr6 = LSTM(int(lstm_units/4), 
                activation=hid_lyr_act, 
                input_shape=mem_shape, 
                return_sequences=rs, 
                name="TxtDisLSTM_6")(lyr5)

    # batch normalization + drop layers to avoid overfit
    lyr7 = BatchNormalization(name="TxtDisBN_7")(lyr6)
    lyr8 = Dropout(hid_ldrop, name="TxtDisDrop_8")(lyr7)

    # flatten from 2D to 1D
    lyr9 = Flatten(name="TxtDisFlat_9")(lyr8)

    # dense classifier layers
    lyr10 = Dense(int(mid_disn), activation=hid_cls_act, name="TxtDisDense_10")(lyr9)
    lyr11 = Dense(int(mid_disn/2), activation=hid_cls_act, name="TxtDisDense_11")(lyr10)
    # drop layer
    lyr12 = Dropout(hid_ldrop, name="TxtDisDrop_12")(lyr11)

    # dense classifier layers
    lyr13 = Dense(int(mid_disn/4), activation=hid_cls_act, name="TxtDisDense_13")(lyr12)
    lyr14 = Dense(int(mid_disn/8), activation=hid_cls_act, name="TxtDisDense_14")(lyr13)
    # drop layer
    lyr15 = Dropout(hid_ldrop, name="TxtDisDrop_15")(lyr14)

    # dense classifier layers
    lyr16 = Dense(int(mid_disn/16), activation=hid_cls_act, name="TxtDisDense_16")(lyr15)
    lyr17 = Dense(int(mid_disn/32), activation=hid_cls_act, name="TxtDisDense_17")(lyr16)

    # output layer
    out_cls = Dense(out_nsize, activation=out_lyr_act, name="TxtDisOut")(lyr17)

    # MODEL DEFINITION
    model = Model(inputs=in_txt, outputs=out_cls)
    return model


# %%
def create_txt_gan(gen_model, dis_model, gan_cfg):

    # getting GAN Config
    ls = gan_cfg.get("loss")
    opt = gan_cfg.get("optimizer")
    met = gan_cfg.get("metrics")

    # make weights in the discriminator not trainable
    dis_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise = gen_model.input
    # get image output from the generator model
    gen_output = gen_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = dis_model(gen_output)
    # define gan model as taking noise and label and outputting a classification
    model = Model(gen_noise, gan_output)
    # compile model
    model.compile(loss=ls, optimizer=opt, metrics=met)

    return model

# %% [markdown]
# ## Conditional Img GAN: CGAN-img

# %%
# convolutional generator for images
def create_img_cgenerator(latent_shape, n_labels, model_cfg):

    # MODEL CONFIG
    # config for conditional labels
    memory = latent_shape[0]
    features = latent_shape[1]
    lbl_neurons = model_cfg.get("labels_neurons")
    lbl_ly_actf = model_cfg.get("labels_lyr_activation")
    hid_ldrop = model_cfg.get("gen_dropout_rate")

    # def of the latent space size for the input
    # input layer config, latent txt space
    latent_n = model_cfg.get("latent_img_size")
    in_lyr_act = model_cfg.get("input_lyr_activation")
    # latent img shape
    latent_img_shape = model_cfg.get("latent_img_shape")

    # hidden layer config
    filters = model_cfg.get("filters")
    ksize = model_cfg.get("kernel_size")
    stsize = model_cfg.get("stride")
    pad = model_cfg.get("padding")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    hid_ldrop = model_cfg.get("gen_dropout_rate")
    mval = model_cfg.get("mask_value")
    rs = model_cfg.get("return_sequences")
    lstm_units = model_cfg.get("lstm_neurons")

    # output layer condig
    out_filters = model_cfg.get("output_filters")
    out_ksize = model_cfg.get("output_kernel_size")
    out_stsize = model_cfg.get("output_stride")
    out_pad = model_cfg.get("output_padding")
    img_shape = model_cfg.get("output_shape")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # CONDITIONAL LABELS LAYERS
    # label input
    in_labels = Input(shape=(n_labels,), name="ImgCGenLblIn")
    # embedding categorical textual input
    cond1 = Embedding(memory, features, input_length=n_labels, name="ImgCGenLblEmb_1")(in_labels)

    # flat layer
    cond2 = Flatten(name="ImgCGenLblFlat_2")(cond1)
    # dense layers
    cond3 = Dense(int(lbl_neurons/2), activation=lbl_ly_actf, name="ImgCGenLblDense_3")(cond2)

    # batch normalization + drop layers to avoid overfit
    cond4 = BatchNormalization(name="ImgCGenLblBN_4")(cond3)
    cond5 = Dropout(hid_ldrop, name="ImgCGenLblDrop_5")(cond4)

    cond6 = Dense(lbl_neurons, activation=lbl_ly_actf, name="ImgCGenLblDense_6")(cond5)
    # reshape layer
    cond7 = Reshape(latent_shape, name="ImgCGenLblOut")(cond6)

    # GENERATOR DEFINITION
    # LAYER CREATION
    # input layer
    in_latent = Input(shape=latent_shape, name="ImgCGenIn")

    # concat generator layers + label layers
    lbl_concat = Concatenate(axis=-1, name="ImgCGenConcat")([in_latent, cond7])

    # masking input text
    lyr1 = Masking(mask_value=mval, input_shape=latent_shape, 
                    name = "ImgCGenMask_1")(lbl_concat) # contat!!!!

    # intermediate recurrent layer
    lyr2 = LSTM(lstm_units, activation=in_lyr_act, 
                    input_shape=latent_shape, 
                    return_sequences=rs, name="ImgCGenLSTM_2")(lyr1)

    # flatten from 2D to 1D
    lyr3 = Flatten(name="ImgCGenFlat_3")(lyr2)

    # dense layer
    lyr4 = Dense(latent_n, 
                activation=hid_lyr_act, 
                name="ImgCGenDense_4")(lyr3)
    
    # reshape layer 1D-> 2D (rbg image)
    lyr5 = Reshape(latent_img_shape, name="ImgCGenReshape_5")(lyr4)

    # transpose conv2D layer
    lyr6 = Conv2DTranspose(int(filters/8), kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgCGenConv2D_6")(lyr5)

    # batch normalization + drop layers to avoid overfit
    lyr7 = BatchNormalization(name="ImgCGenBN_7")(lyr6)
    lyr8 = Dropout(hid_ldrop, name="ImgCGenDrop_8")(lyr7)


    # transpose conv2D layer
    lyr9 = Conv2DTranspose(int(filters/4), kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgCGenConv2D_9")(lyr8)

    # transpose conv2D layer
    lyr10 = Conv2DTranspose(int(filters/2), kernel_size=ksize, 
                            strides=out_stsize, activation=out_lyr_act, 
                            padding=out_pad, name="ImgCGenConv2D_10")(lyr9)

    # batch normalization + drop layers to avoid overfit
    lyr11 = BatchNormalization(name="ImgCGenBN_11")(lyr10)
    lyr12 = Dropout(hid_ldrop, name="ImgCGenDrop_12")(lyr11)

    # transpose conv2D layer
    lyr13 = Conv2DTranspose(filters, kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgCGenConv2D_13")(lyr12)

    # output layer
    out_img = Conv2D(out_filters, kernel_size=out_ksize, 
                        strides=out_stsize, activation=out_lyr_act, 
                        padding=out_pad, input_shape=img_shape, 
                        name="ImgCGenOut")(lyr13)

    # MODEL DEFINITION
    model = Model(inputs=[in_latent, in_labels], outputs=out_img)
    return model


# %%
# convolutional discriminator for images
def create_img_cdiscriminator(img_shape, n_labels, model_cfg):

    # MODEL CONFIG
    # config for conditional labels
    memory = model_cfg.get("timesteps")
    features = model_cfg.get("max_features")
    lbl_neurons = model_cfg.get("labels_neurons")
    lbl_ly_actf = model_cfg.get("labels_lyr_activation")
    lbl_filters = model_cfg.get("labels_filters")
    lbl_ksize = model_cfg.get("labels_kernel_size")
    lbl_stsize = model_cfg.get("labels_stride")
    gen_reshape = model_cfg.get("labels_reshape")
    hid_ldrop = model_cfg.get("dis_dropout_rate")

    # input layer config, image classification
    in_lyr_act = model_cfg.get("input_lyr_activation")
    in_filters = model_cfg.get("input_filters")
    in_ksize = model_cfg.get("input_kernel_size")
    in_stsize = model_cfg.get("input_stride")
    in_pad = model_cfg.get("input_padding")

    # hidden layer config
    filters = model_cfg.get("filters")
    ksize = model_cfg.get("kernel_size")
    stsize = model_cfg.get("stride")
    pad = model_cfg.get("padding")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    hid_ldrop = model_cfg.get("dis_dropout_rate")
    # mid neuron size
    mid_disn = model_cfg.get("mid_dis_neurons")
    hid_cls_act = model_cfg.get("dense_cls_activation")

    # output layer condig
    out_nsize = model_cfg.get("output_dis_neurons")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # LABEL IMG LAYERS
    # label inpuy
    in_labels = Input(shape=(n_labels,), 
                        name="ImgCDisLblIn")

    # embedding categorical textual input
    cond1 = Embedding(memory, features, 
                        input_length=n_labels, 
                        name="ImgCDisLblEmb_1")(in_labels)

    # flat layer
    cond2 = Flatten(name="ImgCDisLblFlat_2")(cond1)
    cond3 = Dense(lbl_neurons, activation=lbl_ly_actf, 
                    name="ImgCDisLblDense_3")(cond2)
    
    # reshape layer
    cond4 = Reshape(gen_reshape, name="ImgCDisLblReshape_4")(cond3)

    # transpose conv2D layers
    cond5 = Conv2DTranspose(int(lbl_filters/8), kernel_size=ksize, 
                                strides=stsize, activation=lbl_ly_actf, 
                                padding=pad, name="ImgCDisLblConv2D_5")(cond4)

    # batch normalization + drop layers to avoid overfit
    cond6 = BatchNormalization(name="ImgCDisLblBN_6")(cond5)
    cond7 = Dropout(hid_ldrop, name="ImgCDisLblDrop_7")(cond6)

    # trnaspose conv2D layers
    cond8 = Conv2DTranspose(int(lbl_filters/4), kernel_size=ksize, 
                                strides=stsize, activation=lbl_ly_actf, 
                                padding=pad, name="ImgCDisLblConv2D_8")(cond7)

    # batch normalization + drop layers to avoid overfit
    cond9 = BatchNormalization(name="ImgCDisLblBN_9")(cond8)
    cond10 = Dropout(hid_ldrop, name="ImgCDisLblDrop_10")(cond9)

    # conditional layer output
    cond11 = Conv2DTranspose(img_shape[2], kernel_size=ksize, 
                                strides=stsize, activation=lbl_ly_actf, 
                                padding=pad, name="ImgCDisLblOut")(cond10)

    # LAYER CREATION
    # input layer
    in_img = Input(shape=img_shape, name="DisImgIn")

    lbl_concat = Concatenate(axis=-1, name="CDisConcat_1")([in_img, cond11])

    # DISCRIMINATOR LAYERS
    # intermediate conv layer
    lyr1 = Conv2D(in_filters, kernel_size=in_ksize, 
                    padding=in_pad, activation=in_lyr_act, 
                    strides=in_stsize, name="ImgCDisConv2D_1")(lbl_concat)

    # intermediate conv layer
    lyr2 = Conv2D(int(filters/2), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgCDisConv2D_2")(lyr1)

    # batch normalization + drop layers to avoid overfit
    lyr3 = BatchNormalization(name="ImgCDisBN_3")(lyr2)
    lyr4 = Dropout(hid_ldrop, name="ImgCDisDrop_4")(lyr3)

    # intermediate conv layer
    lyr5 = Conv2D(int(filters/4), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgCDisConv2D_4")(lyr4)

    # intermediate conv layer
    lyr6 = Conv2D(int(filters/8), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgCDisConv2D_5")(lyr5)

    # batch normalization + drop layers to avoid overfit
    lyr7 = BatchNormalization(name="ImgCDisBN_6")(lyr6)
    lyr8 = Dropout(hid_ldrop, name="ImgCDisDrop_7")(lyr7)

    # flatten from 2D to 1D
    lyr9 = Flatten(name="ImgCDisFlat_8")(lyr8)

    # dense classifier layers
    lyr10 = Dense(int(mid_disn), activation=hid_cls_act, name="ImgCDisDense_9")(lyr9)
    lyr11 = Dense(int(mid_disn/2), activation=hid_cls_act, name="ImgCDisDense_10")(lyr10)
    # drop layer
    lyr12 = Dropout(hid_ldrop, name="ImgCDisDrop_11")(lyr11)

    # dense classifier layers
    lyr13 = Dense(int(mid_disn/4), activation=hid_cls_act, name="ImgCDisDense_12")(lyr12)
    lyr14 = Dense(int(mid_disn/8), activation=hid_cls_act, name="ImgCDisDense_13")(lyr13)
    # drop layer
    lyr15 = Dropout(hid_ldrop, name="ImgCDisDrop_14")(lyr14)

    # dense classifier layers
    lyr16 = Dense(int(mid_disn/16), activation=hid_cls_act, name="ImgCDisDense_15")(lyr15)
    lyr17 = Dense(int(mid_disn/32), activation=hid_cls_act, name="ImgCDisDense_16")(lyr16)

    # output layer
    out_cls = Dense(out_nsize, activation=out_lyr_act, name="ImgCDisOut")(lyr17)

    # MODEL DEFINITION
    model = Model(inputs=[in_img, in_labels], outputs=out_cls)
    return model


# %%
def create_img_cgan(gen_model, dis_model, gan_cfg):

    # getting GAN Config
    ls = gan_cfg.get("loss")
    opt = gan_cfg.get("optimizer")
    met = gan_cfg.get("metrics")

    # make weights in the discriminator not trainable
    dis_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_labels = gen_model.input
    # get image output from the generator model
    gen_output = gen_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = dis_model([gen_output, gen_labels])
    # define gan model as taking noise and label and outputting a classification
    gan_model = Model([gen_noise, gen_labels], gan_output)
    # compile model
    gan_model.compile(loss=ls, optimizer=opt, metrics=met)
    # cgan_model.compile(loss=gan_cfg[0], optimizer=gan_cfg[1])#, metrics=gan_cfg[2])
    return gan_model

# %% [markdown]
# ## Multi GAN txt2img

# %%
# LSTM + Conv discriminator for image and text
def create_multi_discriminator(img_shape, txt_shape, model_cfg):

    # model definition
    model = Model(inputs=[in_img, in_txt], outputs=out_cls)

    return model


# %%
def create_multi_generator(img_shape, txt_shape, model_cfg):

    # model definition
    gen_model = Model(inputs=in_latent, outputs=[out_img, out_txt])
    return model


# %%
def create_multi_gan(gen_model, dis_model, gan_cfg):

    # getting GAN Config
    ls = gan_cfg.get("loss")
    opt = gan_cfg.get("optimizer")
    met = gan_cfg.get("metrics")

    # make weights in the discriminator not trainable
    dis_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_labels = gen_model.input
    # get image output from the generator model
    gen_output = gen_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = dis_model([gen_output, gen_labels])
    # define gan model as taking noise and label and outputting a classification
    model = Model([gen_noise, gen_labels], gan_output)
    # compile model
    model.compile(loss=gan_cfg[0], optimizer=gan_cfg[1], metrics=gan_cfg[2])
    # cgan_model.compile(loss=gan_cfg[0], optimizer=gan_cfg[1])#, metrics=gan_cfg[2])
    return model

# %% [markdown]
# ## Multi CGAN txt2img

# %%
def create_multi_cgenerator(latent_shape, img_shape, txt_shape, n_labels, model_cfg):
    # MODEL CONFIG
    # config for conditional labels
    # print("=======================\n",model_cfg, "=====================")
    memory = latent_shape[0]
    features = latent_shape[1]
    lbl_neurons = model_cfg.get("labels_neurons")
    lbl_ly_actf = model_cfg.get("labels_lyr_activation")
    hid_ldrop = model_cfg.get("gen_dropout_rate")

    # def of the latent space size for the input
    # input layer config, latent txt space
    latent_nimg = model_cfg.get("latent_img_size")
    in_lyr_act = model_cfg.get("input_lyr_activation")
    latent_img_shape = model_cfg.get("latent_img_shape")
    mval = model_cfg.get("mask_value")
    in_rs = model_cfg.get("input_return_sequences")
    in_lstm = model_cfg.get("input_lstm_neurons")

    # hidden layer config
    filters = model_cfg.get("filters")
    ksize = model_cfg.get("kernel_size")
    stsize = model_cfg.get("stride")
    pad = model_cfg.get("padding")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    rs = model_cfg.get("return_sequences")
    lstm_units = model_cfg.get("lstm_neurons")
    latent_ntxt = model_cfg.get("mid_gen_neurons")
    latent_txt_shape = model_cfg.get("latent_lstm_reshape")
    lstm_units = model_cfg.get("lstm_neurons")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    mem_shape = model_cfg.get("memory_shape")
    rs = model_cfg.get("hidden_return_sequences")

    # output layer condig
    out_filters = model_cfg.get("output_filters")
    out_ksize = model_cfg.get("output_kernel_size")
    out_stsize = model_cfg.get("output_stride")
    out_pad = model_cfg.get("output_padding")
    img_shape = model_cfg.get("output_shape")
    out_lyr_act = model_cfg.get("output_lyr_activation")
    txt_shape = model_cfg.get("output_neurons")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # MODEL DEF
    # CONDITIONAL LABELS LAYERS FOR IMG + TXT
    # label input
    in_labels = Input(shape=(n_labels,), name="ImgTxtCGenLblIn")
    # embedding categorical textual input
    cond1 = Embedding(memory, features, input_length=n_labels, name="ImgTxtCGenLblEmb_1")(in_labels)

    # flat layer
    cond2 = Flatten(name="ImgTxtCGenLblFlat_2")(cond1)
    # dense layers
    cond3 = Dense(int(lbl_neurons/2), activation=lbl_ly_actf, name="ImgTxtCGenLblDense_3")(cond2)

    # batch normalization + drop layers to avoid overfit
    cond4 = BatchNormalization(name="ImgTxtCGenLblBN_4")(cond3)
    cond5 = Dropout(hid_ldrop, name="ImgTxtCGenLblDrop_5")(cond4)

    cond6 = Dense(lbl_neurons, activation=lbl_ly_actf, name="ImgTxtCGenLblDense_6")(cond5)
    # reshape layer
    cond7 = Reshape(latent_shape, name="ImgTxtCGenLblOut")(cond6)

    # GENERATOR DEFINITION
    #LATENT LAYER CREATION
    # input layer
    in_latent = Input(shape=latent_shape, name="ImgTxtCGenIn")

    # concat generator layers + label layers
    lbl_concat = Concatenate(axis=-1, name="ImgTxtCGenConcat")([in_latent, cond7])

    # masking input text
    olyr1 = Masking(mask_value=mval, input_shape=latent_shape, 
                    name = "ImgTxtCGenMask_1")(lbl_concat) # contat!!!!

    # intermediate recurrent layer
    olyr2 = LSTM(lstm_units, activation=in_lyr_act, 
                    input_shape=latent_shape, 
                    return_sequences=rs, name="ImgTxtCGenLSTM_2")(olyr1)

    # flatten from 2D to 1D
    olyr3 = Flatten(name="ImgTxtCGenFlat_3")(olyr2)

    # dense layer
    olyr4 = Dense(latent_nimg, 
                activation=hid_lyr_act, 
                name="ImgTxtCGenDense_4")(olyr3)
    
    olyr5 = Dense(latent_ntxt, 
                activation=hid_lyr_act, 
                name="ImgTxtCGenDense_5")(olyr3)

    # IMG GENERATOR
    # reshape layer 1D-> 2D (rbg image)
    ilyr5 = Reshape(latent_img_shape, name="ImgCGenReshape_5")(olyr4)

    # transpose conv2D layer
    ilyr6 = Conv2DTranspose(int(filters/8), kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgCGenConv2D_6")(ilyr5)

    # batch normalization + drop layers to avoid overfit
    ilyr7 = BatchNormalization(name="ImgCGenBN_7")(ilyr6)
    ilyr8 = Dropout(hid_ldrop, name="ImgCGenDrop_8")(ilyr7)


    # transpose conv2D layer
    ilyr9 = Conv2DTranspose(int(filters/4), kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgCGenConv2D_9")(ilyr8)

    # transpose conv2D layer
    ilyr10 = Conv2DTranspose(int(filters/2), kernel_size=ksize, 
                            strides=out_stsize, activation=out_lyr_act, 
                            padding=out_pad, name="ImgCGenConv2D_10")(ilyr9)

    # batch normalization + drop layers to avoid overfit
    ilyr11 = BatchNormalization(name="ImgCGenBN_11")(ilyr10)
    ilyr12 = Dropout(hid_ldrop, name="ImgCGenDrop_12")(ilyr11)

    # transpose conv2D layer
    ilyr13 = Conv2DTranspose(filters, kernel_size=ksize, 
                            strides=stsize, activation=hid_lyr_act, 
                            padding=pad, name="ImgCGenConv2D_13")(ilyr12)

    # output layer
    out_img = Conv2D(out_filters, kernel_size=out_ksize, 
                        strides=out_stsize, activation=out_lyr_act, 
                        padding=out_pad, input_shape=img_shape, 
                        name="ImgCGenOut")(ilyr13)

    # TXT GENERATOR
    # reshape layer 1D-> 2D (descriptive txt)
    tlyr6 = Reshape(latent_txt_shape, name="TxtGenReshape_6")(olyr5)

    # batch normalization + drop layers to avoid overfit
    tlyr7 = BatchNormalization(name="TxtGenBN_7")(tlyr6)
    tlyr8 = Dropout(hid_ldrop, name="TxtGenDrop_8")(tlyr7)

    # intermediate recurrent layer
    tlyr9 = LSTM(int(lstm_units/4), activation=hid_lyr_act, 
                    input_shape=mem_shape, 
                    return_sequences=rs, 
                    name="TxtGenLSTM_10")(tlyr8)

    # intermediate recurrent layer
    tlyr10 = LSTM(int(lstm_units/2), activation=hid_lyr_act, 
                    input_shape=mem_shape, 
                    return_sequences=rs, 
                    name="TxtGenLSTM_11")(tlyr9)

    # batch normalization + drop layers to avoid overfit
    tlyr11 = BatchNormalization(name="TxtGenBN_12")(tlyr10)
    tlyr12 = Dropout(hid_ldrop, name="TxtGenDrop_12")(tlyr11)

    # output layer, dense time sequential layer.
    tlyr13 = LSTM(lstm_units, activation=hid_lyr_act, 
                    input_shape=mem_shape, 
                    return_sequences=rs, 
                    name="TxtGenDrop_13")(tlyr12)

    out_txt = TimeDistributed(Dense(txt_shape, activation=out_lyr_act), name = "GenTxtOut")(tlyr13)

    # MODEL DEFINITION
    model = Model(inputs=[in_latent, in_labels], outputs=[out_img, out_txt])

    return model


# %%
# LSTM + Conv conditianal discriminator for text and images
def create_multi_cdiscriminator(img_shape, txt_shape, n_labels, model_cfg):

    # MODEL CONFIG
    # config for txt + img conditional labels
    memory = model_cfg.get("timesteps")
    features = model_cfg.get("max_features")
    lbl_neurons = model_cfg.get("labels_neurons")
    lbl_ly_actf = model_cfg.get("labels_lyr_activation")
    lbl_filters = model_cfg.get("labels_filters")
    lbl_ksize = model_cfg.get("labels_kernel_size")
    lbl_stsize = model_cfg.get("labels_stride")
    lbl_lstm = model_cfg.get("labels_lstm_neurons")
    lbl_rs = model_cfg.get("labels_return_sequences")
    dis_img_reshape = model_cfg.get("labels_img_reshape")
    dis_txt_reshape = model_cfg.get("labels_txt_reshape")

    # input layer config for image classification
    in_lyr_act = model_cfg.get("input_lyr_activation")
    in_filters = model_cfg.get("input_filters")
    in_ksize = model_cfg.get("input_kernel_size")
    in_stsize = model_cfg.get("input_stride")
    in_pad = model_cfg.get("input_padding")

    # input layer config for txt classification
    mval = model_cfg.get("mask_value")
    in_rs = model_cfg.get("input_return_sequences")
    in_lstm = model_cfg.get("input_lstm_neurons")
    in_lyr_act = model_cfg.get("input_lyr_activation")

    # encoding hidden layer config for image classification
    filters = model_cfg.get("filters")
    ksize = model_cfg.get("kernel_size")
    stsize = model_cfg.get("stride")
    pad = model_cfg.get("padding")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")
    hid_ldrop = model_cfg.get("dis_dropout_rate")

    # encoding hidden layer config for text classification
    lstm_units = model_cfg.get("lstm_neurons")
    mem_shape = model_cfg.get("memory_shape")
    rs = model_cfg.get("hidden_return_sequences")

    # mid classification config
    mid_disn = model_cfg.get("mid_dis_neurons")
    hid_cls_act = model_cfg.get("dense_cls_activation")

    # output layer config
    out_nsize = model_cfg.get("output_dis_neurons")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # IMG/TXT LABELS CONDITIONAL LAYERS
    # labels input
    in_labels = Input(shape=(n_labels,), 
                        name="MultiCDisLblIn")

    # embedding categorical textual input
    cond1 = Embedding(memory, features, 
                        input_length=n_labels, 
                        name="MultiCDisLblEmb_1")(in_labels)

    # flat layer
    cond2 = Flatten(name="MultiCDisLblFlat_2")(cond1)
    # img dense layer
    cond3 = Dense(lbl_neurons, activation=lbl_ly_actf, 
                    name="MultiCDisLblDense_3")(cond2)
    
    # image reshape layer
    cond4i = Reshape(dis_img_reshape, name="ImgCDisLblReshape_4")(cond3)

    # txt dense layer
    cond5 = Dense(int(memory*features), activation=lbl_ly_actf, 
                    name="MultiCDisLblDense_5")(cond2)
    # txt reshape layer
    cond6t = Reshape(dis_txt_reshape, name="TxtCDisLblReshape_6")(cond5)

    # transpose conv2D layer for img
    cond7 = Conv2DTranspose(int(lbl_filters/8), kernel_size=ksize, 
                                strides=stsize, activation=lbl_ly_actf, 
                                padding=pad, name="ImgCDisLblConv2D_7")(cond4i)

    # batch normalization + drop layers to avoid overfit for img
    cond8 = BatchNormalization(name="ImgCDisLblBN_8")(cond7)
    cond9 = Dropout(hid_ldrop, name="ImgCDisLblDrop_9")(cond8)

    # transpose conv2D layers for img
    cond10 = Conv2DTranspose(int(lbl_filters/4), kernel_size=ksize, 
                                strides=stsize, activation=lbl_ly_actf, 
                                padding=pad, name="ImgCDisLblConv2D_10")(cond9)

    # batch normalization + drop layers to avoid overfit for img
    cond11 = BatchNormalization(name="ImgCDisLblBN_11")(cond10)
    cond12 = Dropout(hid_ldrop, name="ImgCDisLblDrop_12")(cond11)

    # conditional layer output for img
    cond13 = Conv2DTranspose(img_shape[2], kernel_size=ksize, 
                                strides=stsize, activation=lbl_ly_actf, 
                                padding=pad, name="ImgCDisLblOut")(cond12)

    # intermediate LSTM layer for text
    cond14 = LSTM(int(lbl_lstm/2), 
                activation=lbl_ly_actf, 
                input_shape=mem_shape, 
                return_sequences=lbl_rs, 
                name="TxtCDisLblLSTM_14")(cond6t)

    # batch normalization + drop layers to avoid overfit for img
    cond15 = BatchNormalization(name="TxtCDisLblBN_15")(cond14)
    cond16 = Dropout(hid_ldrop, name="TxtCDisLblDrop_16")(cond15)

    # intermediate LSTM layer for text
    cond17 = LSTM(lbl_lstm, 
                activation=lbl_ly_actf, 
                input_shape=mem_shape, 
                return_sequences=lbl_rs, 
                name="TxtCDisLblLSTM_17")(cond16)

    # LAYER CREATION
    # input layer
    in_img = Input(shape=img_shape, name="CDisImgIn")

    concat_img = Concatenate(axis=-1, name="ImgCDisConcat_19")([in_img, cond13])

    # DISCRIMINATOR LAYERS
    # intermediate conv layer
    lyr1 = Conv2D(in_filters, kernel_size=in_ksize, 
                    padding=in_pad, activation=in_lyr_act, 
                    strides=in_stsize, name="ImgCDisConv2D_20")(concat_img)

    # intermediate conv layer
    lyr2 = Conv2D(int(filters/2), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgCDisConv2D_21")(lyr1)

    # batch normalization + drop layers to avoid overfit
    lyr3 = BatchNormalization(name="ImgCDisBN_22")(lyr2)
    lyr4 = Dropout(hid_ldrop, name="ImgCDisDrop_23")(lyr3)

    # intermediate conv layer
    lyr5 = Conv2D(int(filters/4), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgCDisConv2D_24")(lyr4)

    # intermediate conv layer
    lyr6 = Conv2D(int(filters/8), kernel_size=ksize, 
                    padding=pad, activation=hid_lyr_act, 
                    strides=stsize, name="ImgCDisConv2D_25")(lyr5)

    # batch normalization + drop layers to avoid overfit
    lyr7 = BatchNormalization(name="ImgCDisBN_26")(lyr6)
    lyr8 = Dropout(hid_ldrop, name="ImgCDisDrop_27")(lyr7)

    # flatten from 2D to 1D
    lyr9 = Flatten(name="ImgCDisFlat_28")(lyr8)

    #TXT DISCRIMINATOR
    # LAYER CREATION
    # input layer
    in_txt = Input(shape=txt_shape, name="CDisTxtIn")

    # concat txt input with labels conditional
    concat_txt = Concatenate(axis=-1, name="TxtCDisConcat_30")([in_txt, cond17])

    # DISCRIMINATOR LAYERS
    # masking input text
    lyr10 = Masking(mask_value=mval, input_shape=txt_shape, 
                    name = "TxtCDisMask_31")(concat_txt) # concat1

    # input LSTM layer
    lyr11 = LSTM(in_lstm, activation=in_lyr_act, 
                    input_shape=txt_shape, 
                    return_sequences=in_rs, 
                    name="TxtCDisLSTM_32")(lyr10)

    # batch normalization + drop layers to avoid overfit
    lyr12 = BatchNormalization(name="TxtCDisBN_33")(lyr11)
    lyr13 = Dropout(hid_ldrop, name="TxCtDisDrop_34")(lyr12)

    # intermediate LSTM layer
    lyr14 = LSTM(int(lstm_units/2), 
                activation=hid_lyr_act, 
                input_shape=mem_shape, 
                return_sequences=rs, 
                name="TxtCDisLSTM_35")(lyr13)

    # intermediate LSTM layer
    lyr15 = LSTM(int(lstm_units/4), 
                activation=hid_lyr_act, 
                input_shape=mem_shape, 
                return_sequences=rs, 
                name="TxtCDisLSTM_36")(lyr14)

    # batch normalization + drop layers to avoid overfit
    lyr16 = BatchNormalization(name="TxtCDisBN_37")(lyr15)
    lyr17 = Dropout(hid_ldrop, name="TxtCDisDrop_38")(lyr16)

    # flatten from 2D to 1D
    lyr18 = Flatten(name="TxtCDisFlat_39")(lyr17)

    # concat img encoding + txt encoding
    concat_encoding = Concatenate(axis=-1, name="MultiCDisConcat_40")([lyr18, lyr9])

    # dense classifier layers
    lyr19 = Dense(int(mid_disn), activation=hid_cls_act, name="MultiCDisDense_41")(concat_encoding)
    lyr20 = Dense(int(mid_disn/2), activation=hid_cls_act, name="MultiCDisDense_42")(lyr19)
    # drop layer
    lyr21 = Dropout(hid_ldrop, name="MultiCDisDrop_43")(lyr20)

    # dense classifier layers
    lyr22 = Dense(int(mid_disn/4), activation=hid_cls_act, name="MultiCDisDense_44")(lyr21)
    lyr23 = Dense(int(mid_disn/8), activation=hid_cls_act, name="MultiCDisDense_45")(lyr22)
    # drop layer
    lyr24 = Dropout(hid_ldrop, name="MultiCDisDrop_46")(lyr23)

    # dense classifier layers
    lyr25 = Dense(int(mid_disn/16), activation=hid_cls_act, name="MultiCDisDense_47")(lyr24)
    lyr26 = Dense(int(mid_disn/32), activation=hid_cls_act, name="MultiCDisDense_48")(lyr25)

    # output layer
    out_cls = Dense(out_nsize, activation=out_lyr_act, name="MultiCDisOut")(lyr26)

    # model definition
    model = Model(inputs=[in_img, in_txt, in_labels], outputs=out_cls)

    return model


# %%
def create_multi_cgan(gen_model, dis_model, gan_cfg):

    # getting GAN Config
    ls = gan_cfg.get("loss")
    opt = gan_cfg.get("optimizer")
    met = gan_cfg.get("metrics")

    # make weights in the discriminator not trainable
    dis_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_labels = gen_model.input
    # get image output from the generator model
    gen_img, gen_txt = gen_model.output
    # connect image output and label input from generator as inputs to discriminator
    gan_output = dis_model([gen_img, gen_txt, gen_labels])
    # define gan model as taking noise and label and outputting a classification
    gan_model = Model([gen_noise, gen_labels], gan_output)
    # compile model
    gan_model.compile(loss=ls, optimizer=opt, metrics=met)
    # cgan_model.compile(loss=gan_cfg[0], optimizer=gan_cfg[1])#, metrics=gan_cfg[2])
    return gan_model

# %% [markdown]
# ## ML Models Configuration
# ### GAN-img definition

# %%
# img generator config
img_gen_cfg = {
    "mask_value": 0.0,
    "return_sequences": True,
    "lstm_neurons": 400,
    "latent_img_size": 50*50*3,
    "input_lyr_activation": "relu",
    "latent_img_shape": (50,50,3),
    "filters": 64, 
    "kernel_size": (4,4),
    "stride": (2,2),
    "padding": "same",
    "hidden_lyr_activation": LeakyReLU(alpha=0.2),
    "gen_dropout_rate": 0.3,
    "output_filters": img_og_shape[2],
    "output_kernel_size": (3,3),
    "output_stride": (1,1),
    "output_padding": "same",
    "output_shape": X_img[0].shape,
    "output_lyr_activation": "tanh",
    }

print("GAN-img Generator Config:\n", img_gen_cfg)


# %%
# img discriminator config
img_dis_cfg = {
    "input_lyr_activation": "relu",
    "input_filters": 64,
    "input_kernel_size": (4,4),
    "input_stride": (2,2),
    "input_padding": "same",
    "filters": 64,
    "kernel_size": (4,4),
    "stride": (2,2),
    "padding": "same",
    "hidden_lyr_activation": LeakyReLU(alpha=0.2),
    "dis_dropout_rate": 0.2,
    "mid_dis_neurons": 50*50*2,
    "dense_cls_activation": LeakyReLU(alpha=0.2),
    "output_dis_neurons": 1,
    "output_lyr_activation": "sigmoid",
    "loss": "binary_crossentropy",
    "optimizer": Adam(learning_rate=0.0004, beta_1=0.5),
    "metrics": ["accuracy"],
    }

print("GAN-img Discriminator Config:\n", img_dis_cfg)


# %%
# img GAN config
gan_cfg = {
    "loss": "binary_crossentropy",
    "optimizer": Adam(learning_rate=0.0002, beta_1=0.5),
    "metrics": ["accuracy"],
    }

print("GAN-img Config:\n", gan_cfg)

# %% [markdown]
# ### GAN-txt definition

# %%
# txt generator config
txt_gen_cfg = {
    "mask_value": 0.0,
    "input_return_sequences": True,
    "input_lstm_neurons": 400,
    "input_lyr_activation": "relu",
    "mid_gen_neurons": timesteps*X_txt.shape[2],
    "lstm_neurons": 400,
    "hidden_lyr_activation": LeakyReLU(alpha=0.2),
    "hidden_return_sequences": True,
    "gen_dropout_rate": 0.3,
    "latent_lstm_reshape": X_txt[0].shape,
    "memory_shape": X_txt[0].shape,
    "output_neurons": X_txt.shape[2],
    "output_shape": X_txt[0].shape,
    "output_lyr_activation": "tanh",
    "output_return_sequences": True,
    }

print("GAN-txt Generator Config:\n", txt_gen_cfg)


# %%
# txt discriminator config
txt_dis_cfg = {
    "mask_value": 0.0,
    "input_return_sequences": True,
    "input_lstm_neurons": 400,
    "input_lyr_activation": "relu",
    "lstm_neurons": 400,
    "hidden_lyr_activation": LeakyReLU(alpha=0.2),
    "hidden_return_sequences": True,
    "hidden_lyr_activation": LeakyReLU(alpha=0.2),
    "memory_shape": X_txt[0].shape,
    "dis_dropout_rate": 0.2,
    "mid_dis_neurons": timesteps*X_txt.shape[2],
    "dense_cls_activation": LeakyReLU(alpha=0.2),
    "output_dis_neurons": 1,
    "output_lyr_activation": "sigmoid",
    "loss": "binary_crossentropy",
    "optimizer": Adam(learning_rate=0.0004, beta_1=0.5),
    "metrics": ["accuracy"],
    }

print("GAN-txt Discriminator Config:\n", txt_dis_cfg)

# %% [markdown]
# ### CGAN-img definition

# %%
img_cgen_cfg = {
    "mask_value": 0.0,
    "return_sequences": True,
    "lstm_neurons": 500,
    "latent_img_size": 50*50*3,
    "input_lyr_activation": "relu",
    "latent_img_shape": (50,50,3),
    "filters": 128, 
    "kernel_size": (4,4),
    "stride": (2,2),
    "padding": "same",
    "hidden_lyr_activation": LeakyReLU(alpha=0.2),
    "gen_dropout_rate": 0.3,
    "output_filters": img_og_shape[2],
    "output_kernel_size": (3,3),
    "output_stride": (1,1),
    "output_padding": "same",
    "output_shape": X_img[0].shape,
    "output_lyr_activation": "tanh",
    "labels_neurons": timesteps*X_txt.shape[2],
    "labels_lyr_activation": LeakyReLU(alpha=0.2),
    }

print("CGAN-img Generator Config:\n", img_cgen_cfg)


# %%
img_cdis_cfg = {
    "input_lyr_activation": "relu",
    "input_filters": 128,
    "input_kernel_size": (4,4),
    "input_stride": (2,2),
    "input_padding": "same",
    "filters": 128,
    "kernel_size": (4,4),
    "stride": (2,2),
    "padding": "same",
    "hidden_lyr_activation": LeakyReLU(alpha=0.2),
    "dis_dropout_rate": 0.2,
    "mid_dis_neurons": 50*50*2,
    "dense_cls_activation": LeakyReLU(alpha=0.2),
    "output_dis_neurons": 1,
    "output_lyr_activation": "sigmoid",
    "labels_lyr_activation": LeakyReLU(alpha=0.2),
    "timesteps": timesteps,
    "max_features": X_txt.shape[2],
    "labels_neurons": 50*50*3,
    "labels_lyr_activation": LeakyReLU(alpha=0.2),
    "labels_filters": 64,
    "labels_kernel_size": (4,4),
    "labels_stride": (2,2),
    "labels_reshape": (50,50,3),
    "loss": "binary_crossentropy",
    "optimizer": Adam(learning_rate=0.0004, beta_1=0.5),
    "metrics": ["accuracy"],
    }

print("CGAN-img Generator Config:\n", img_cdis_cfg)


# %%
# txt GAN config
img_cgan_cfg = {
    "loss": "binary_crossentropy",
    "optimizer": Adam(learning_rate=0.0002, beta_1=0.5),
    "metrics": ["accuracy"],
    }

print("CGAN-img Config:\n", img_cgan_cfg)

# %% [markdown]
# ### CGAN-txt2img definition

# %%
multi_cgen_cfg = dict()
multi_cgen_cfg.update(img_cgen_cfg)
multi_cgen_cfg.update(txt_gen_cfg)

print("Multi CGen-txt2img Config:\n", multi_cgen_cfg)


# %%
multi_cdis_cfg = dict()
multi_cdis_cfg.update(img_cdis_cfg)
multi_cdis_cfg.update(txt_dis_cfg)

mcdis_cfg_update = {
    "labels_lstm_neurons": 500,
    "labels_return_sequences": True,
    "labels_img_reshape": (50,50,3),
    "labels_txt_reshape": X_txt[0].shape,
    }

multi_cdis_cfg.update(mcdis_cfg_update)

print("Multi CDis-txt2img Config:\n", multi_cdis_cfg)


# %%
# txt2img CGAN config
multi_cgan_cfg = {
    "loss": "binary_crossentropy",
    "optimizer": Adam(learning_rate=0.00015, beta_1=0.5),
    "metrics": ["accuracy"],
    }

print("Multi CGAN-txt2img Config:\n", multi_cgan_cfg)

# %% [markdown]
# ## ML Model Creation
# ### GAN img definition

# %%
latent_shape = X_txt[0].shape
gen_model = create_img_generator(latent_shape, img_gen_cfg)
print("GAN-img Generator Definition")
# dis_model = Sequential(slim_dis_layers)
gen_model.model_name = "GAN-img Generator"

# DONT compile model
# cdis_model.trainable = False
gen_model.summary()


# %%
img_shape = X_img[0].shape
dis_model = create_img_discriminator(img_shape, img_dis_cfg)
print("GAN-img Discriminator Definition")
# dis_model = Sequential(slim_dis_layers)
dis_model.model_name = "GAN-img Discriminator"

# compile model
dis_model.compile(loss=img_dis_cfg["loss"], 
                    optimizer=img_dis_cfg["optimizer"], 
                    metrics=img_dis_cfg["metrics"])

# cdis_model.trainable = False
dis_model.summary()


# %%
print("GAN-img Model definition")
gan_model = create_img_gan(gen_model, dis_model, gan_cfg)
gan_model.model_name = "GAN-img"
gan_model.summary()


# %%
# saving model topology into png files
print(timestamp)
export_model(gen_model, model_fn_path, gen_model.model_name, timestamp)
export_model(dis_model, model_fn_path, dis_model.model_name, timestamp)
export_model(gan_model, model_fn_path, gan_model.model_name, timestamp)

# %% [markdown]
# ### GAN txt definition

# %%
gen_txt_model = create_txt_generator(latent_shape, txt_gen_cfg)
print("GAN-txt Generator Definition")
# dis_model = Sequential(slim_dis_layers)
gen_txt_model.model_name = "GAN-txt Generator"

# DONT compile model
# cdis_model.trainable = False
gen_txt_model.summary()


# %%
txt_shape = X_txt[0].shape
print(txt_shape)
dis_txt_model = create_txt_discriminator(txt_shape, txt_dis_cfg)
print("GAN-txt Discriminator Definition")
# dis_model = Sequential(slim_dis_layers)
dis_txt_model.model_name = "GAN-txt Discriminator"

# compile model
dis_txt_model.compile(loss=txt_dis_cfg["loss"], 
                    optimizer=txt_dis_cfg["optimizer"], 
                    metrics=txt_dis_cfg["metrics"])

# cdis_model.trainable = False
dis_txt_model.summary()


# %%
print("GAN-txt Model definition")
gan_txt_model = create_img_gan(gen_txt_model, dis_txt_model, gan_cfg)
gan_txt_model.summary()
gan_txt_model.model_name = "GAN-txt"


# %%
# saving model topology into png files
print(timestamp)
export_model(gen_txt_model, model_fn_path, gen_txt_model.model_name, timestamp)
export_model(dis_txt_model, model_fn_path, dis_txt_model.model_name, timestamp)
export_model(gan_txt_model, model_fn_path, gan_txt_model.model_name, timestamp)


# %%
n_labels = y_labels[0].shape[0]
print(n_labels)
cgen_img_model = create_img_cgenerator(latent_shape, n_labels, img_cgen_cfg)
print("CGAN-img Generator Definition")
# dis_model = Sequential(slim_dis_layers)
cgen_img_model.model_name = "CGAN-img Generator"

# DONT compile model
# cdis_model.trainable = False
cgen_img_model.summary()


# %%
img_shape = X_img[0].shape
cdis_img_model = create_img_cdiscriminator(img_shape, n_labels, img_cdis_cfg)
print("CGAN-img Discriminator Definition")
# dis_model = Sequential(slim_dis_layers)
cdis_img_model.model_name = "CGAN-img Discriminator"

# compile model
cdis_img_model.compile(loss=img_cdis_cfg["loss"], 
                    optimizer=img_cdis_cfg["optimizer"], 
                    metrics=img_cdis_cfg["metrics"])

# cdis_model.trainable = False
cdis_img_model.summary()


# %%
print("CGAN-img Model definition")
cgan_img_model = create_img_cgan(cgen_img_model, cdis_img_model, gan_cfg)
cgan_img_model.summary()
cgan_img_model.model_name = "CGAN-img"


# %%
# saving model topology into png files
print(timestamp)
export_model(gen_txt_model, model_fn_path, gen_txt_model.model_name, timestamp)
export_model(cdis_img_model, model_fn_path, cdis_img_model.model_name, timestamp)
export_model(cgan_img_model, model_fn_path, cgan_img_model.model_name, timestamp)

# %% [markdown]
# ### Multi CGAN-txt2img

# %%
multi_cgen_model = create_multi_cgenerator(latent_shape, img_shape, txt_shape, n_labels, multi_cgen_cfg)
print("Multi CGAN-txt2img Generator Definition")
# dis_model = Sequential(slim_dis_layers)
multi_cgen_model.model_name = "Multi CGAN-txt2img Generator"

# DONT compile model
# cdis_model.trainable = False
multi_cgen_model.summary()


# %%
multi_cdis_model = create_multi_cdiscriminator(img_shape, txt_shape, n_labels, multi_cdis_cfg)
print("Multi CGAN-txt2img Discriminator Definition")
# dis_model = Sequential(slim_dis_layers)
multi_cdis_model.model_name = "Multi CGAN-txt2img Discriminator"
# compile model

multi_cdis_model.compile(loss=multi_cdis_cfg["loss"], 
                    optimizer=multi_cdis_cfg["optimizer"], 
                    metrics=multi_cdis_cfg["metrics"])

# compile model
multi_cdis_model.summary()


# %%
print("Multi CGAN-txt2img Model definition")
multi_cgan_model = create_multi_cgan(multi_cgen_model, multi_cdis_model, gan_cfg)
multi_cgan_model.summary()
multi_cgan_model.model_name = "Multi CGAN-txt2img"


# %%
# saving model topology into png files
print(timestamp)
export_model(multi_cgen_model, model_fn_path, multi_cgen_model.model_name, timestamp)
export_model(multi_cdis_model, model_fn_path, multi_cdis_model.model_name, timestamp)
export_model(multi_cgan_model, model_fn_path, multi_cgan_model.model_name, timestamp)


# %%
print("-Images:", X_img.shape, "\n-Text:", X_txt.shape, "\n-Real/Fake:", y.shape, "\n-txt&img Labels:", y_labels.shape)


# %%
# training and batch size
gan_train_cfg = {
    "epochs":40,
    "batch_size":32,
    "synth_batch": 1,
    "balance_batch": False,
    "gen_sample_size": 3,
    "models_fn_path": model_fn_path,
    "report_fn_path": report_fn_path,
    "dis_model_name": multi_cgen_model.model_name,
    "gen_model_name": multi_cdis_model.model_name,
    "gan_model_name": multi_cgan_model.model_name,
    "check_epochs": 20,
    "save_epochs": 60,
    "max_save_models": 5,
    "latent_shape": X_txt[0].shape,
    "pretrained": False,
    "conditioned": True,
    "dataset_size": X_img.shape[0],
    "img_shape": X_img[0].shape,
    "txt_shape": X_txt[0].shape,
    "label_shape": y_labels[0].shape,
    "cat_shape": y[0].shape,
    # "data_cols": 2,
    # "data_cols": 3,
    "data_cols": 4,
    }

print("Model Training Config:\n", gan_train_cfg)


# %%
# gan_data = (X_img, y)
# gan_data = (X_img, y_labels, y)
gan_data = (X_img, X_txt, y_labels, y)
print(X_img.shape, X_txt.shape, y_labels.shape, y.shape)
print(len(gan_data))


# %%
# traininng with the traditional gan
# training_gan(gen_model, dis_model, gan_model, gan_data, gan_train_cfg)

# training with the conditional gan with images
# training_gan(cgen_img_model, cdis_img_model, cgan_img_model, gan_data, gan_train_cfg)

# training with the muti conditional gan with images + text
training_model(multi_cgen_model, multi_cdis_model, multi_cgan_model, gan_data, gan_train_cfg)


# %%



# %%



# %%



