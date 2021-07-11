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
import os
import re
import time
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
from wordcloud import WordCloud

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
# tf.config.experimental.set_per_process_memory_fraction(0.75)
# tf.config.gpu.set_per_process_memory_growth(True)
from tensorflow.python.client import device_lib
from keras import backend as K
# from tensorflow.keras.layers

# preprocessing and processing
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.callbacks import EarlyStopping

# models
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model 

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

# # FUNCTION DEFINITION
# fuction for GPU memory formatting
def sizeof_fmt(num, suffix='B'):
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, 'Yi', suffix)


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


# function to read the image from file with cv2
def read_img(img_fpn):
    ans = cv2.imread(img_fpn, cv2.IMREAD_UNCHANGED)
    return ans


# fuction to scale the image and reduce cv2
def scale_img(img, scale_pct):

    width = int(img.shape[1]*scale_pct/100)
    height = int(img.shape[0]*scale_pct/100)
    dim = (width, height)
    # resize image
    ans = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return ans


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


# function than rotates the original image to create a new example
# TODO need to correct this
def syth_rgb_img(data):

    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(rotation_range=90)
    ans = datagen.flow(samples, batch_size=1)
    ans = ans[0].astype("uint8")
    ans = np.squeeze(ans, 0)
    return ans

# function to balance dataset
# TODO complete function


def balance_samples(data):
    pass


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

# format the pandas df data into usable word dense vector representation, YOU NEED IT FOR THE CSV to be useful!
def format_dvector(work_corpus):

    ans = list()
    for dvector in work_corpus:
        dvector = eval(dvector)
        dvector = np.asarray(dvector)
        ans.append(dvector)
    ans = np.asarray(ans, dtype="object")
    return ans


# funct to concatenate all label columns into one for a single y in ML training, returns a list
def concat_labels(row, cname):

    ans = list()
    for c in cname:
        r = row[c]
        r = eval(r)
        ans = ans + r

    return ans


# function to save the ML model
def save_model(model, m_path, m_file):

    fpn = os.path.join(m_path, m_file)
    fpn = fpn + ".h5"
    # print(fpn)
    model.save(fpn)


# function to load the ML model
def load_model(model, m_path, m_file):

    fpn = os.path.join(m_path, m_file)
    fpn = fpn + ".h5"
    # model = keras.models.load_model()
    # model = load_model(fpn)
    model.load_weights(fpn)
    # return model


# function to cast dataframe and avoid problems with keras
def cast_batch(data):

    cast_data = list()

    if len(data) >= 2:

        for d in data:
            d = np.asarray(d).astype("float32")
            cast_data.append(d)

    return cast_data


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


# function to create fake elements to train the discriminator
def gen_fake_samples(gen_model, dataset_shape, half_batch):

    # fake data
    fake_data = None
    # conditional labels for the gan model
    conditional = dataset_shape.get("conditioned")
    # configuratin keys for the generator
    latent_dims = dataset_shape.get("latent_dims")
    cat_shape = dataset_shape.get("cat_shape")
    label_shape = dataset_shape.get("label_shape")
    data_cols = dataset_shape.get("data_cols")

    # generator config according to the dataset
    # X:images -> y:Real/Fake
    if data_cols == 2:
        # random textual latent space 
        latent_space = gen_latent_space(latent_dims, half_batch)
        # marking the images as fake in all accounts
        y_fake = gen_fake_negclass(cat_shape, half_batch)
        # random generated image from the model
        Xi_fake = gen_model.predict(latent_space)
        # fake samples
        fake_data = (Xi_fake, y_fake)

    # X_img, X_labels(classification), y (fake/real)
    elif (conditional == True) and data_cols == 3:
        # random textual latent space 
        latent_space = gen_latent_space(latent_dims, half_batch)
        # marking the images as fake in all accounts
        y_fake = gen_fake_negclass(cat_shape, half_batch)
        # marking all the images with fake labels
        Xl_fake = gen_fake_labels(label_shape, half_batch)

        # random generated image from the model
        Xi_fake = gen_model.predict([latent_space, Xl_fake])
        # fake samples
        fake_data = (Xi_fake, Xl_fake, y_fake)

    elif (conditional == False) and data_cols == 3:
        
        # random textual latent space 
        latent_space = gen_latent_space(latent_dims, half_batch)
        # marking the images as fake in all accounts
        y_fake = gen_fake_negclass(cat_shape, half_batch)
        # random generated image + text from the model
        Xi_fake, Xt_fake = gen_model.predict(latent_space)
        # fake samples
        fake_data = (Xi_fake, Xt_fake, y_fake)

    # X_img(rgb), X_txt(text), X_labels(classification), y (fake/real)
    elif data_cols == 4:

        # random textual latent space 
        latent_space = gen_latent_space(latent_dims, half_batch)
        # marking the images as fake in all accounts
        y_fake = gen_fake_negclass(cat_shape, half_batch)
        # marking all the images with fake labels
        Xl_fake = gen_fake_labels(label_shape, half_batch)

        # random generated image from the model
        Xi_fake, Xt_fake = gen_model.predict([latent_space, Xl_fake])
        # fake samples 
        fake_data = (Xi_fake, Xt_fake, Xl_fake, y_fake)

    # casting data type
    fake_data = cast_batch(fake_data)
    
    return fake_data


# function to create inputs to updalte the GAN generator
def gen_latent_data(dataset_shape, batch_size):

    # latent data
    latent_data = None

    # conditional labels for the gan model
    conditional = dataset_shape.get("conditioned")
    # configuratin keys for the generator
    latent_dims = dataset_shape.get("latent_dims")
    cat_shape = dataset_shape.get("cat_shape")
    label_shape = dataset_shape.get("label_shape")
    data_cols = dataset_shape.get("data_cols")

    # generator config according to the dataset
    # X:images -> y:Real/Fake
    if data_cols == 2:
        # random textual latent space 
        latent_space = gen_latent_space(latent_dims, batch_size)
        # marking the images as fake in all accounts
        y_gen = gen_fake_posclass(cat_shape, batch_size)
        # fake samples
        latent_data = (latent_space, y_gen)

    # X_img, X_labels(classification), y (fake/real)
    elif data_cols == 3 and (conditional == True):
        # random textual latent space 
        latent_space = gen_latent_space(latent_dims, batch_size)
        # marking the images as fake in all accounts
        y_gen = gen_fake_posclass(cat_shape, batch_size)
        # marking all the images with fake labels
        Xl_gen = gen_fake_labels(label_shape, batch_size)
        # gen samples
        latent_data = (latent_space, Xl_gen, y_gen)

    elif data_cols == 3 and (conditional == False):
        # random textual latent space 
        latent_space = gen_latent_space(latent_dims, batch_size)
        # marking the images as fake in all accounts
        y_gen = gen_fake_posclass(cat_shape, batch_size)
        # fake samples
        latent_data = (latent_space, y_gen)

    # X_img(rgb), X_txt(text), X_labels(classification), y (fake/real)
    elif data_cols == 4:
        # random textual latent space 
        latent_space = gen_latent_space(latent_dims, batch_size)
        # marking the images as fake in all accounts
        y_gen = gen_fake_posclass(cat_shape, batch_size)
        # marking all the images with fake labels
        Xl_gen = gen_fake_labels(label_shape, batch_size)
        # gen samples
        latent_data = (latent_space, Xl_gen, y_gen)

    return latent_data
# latent_gen = gen_latent_txt(latent_shape, batch_size)
# create inverted category for the fake noisy text
# y_gen = get_fake_positive(cat_shape[0], batch_size)


# function to generate random/latent text for the GAN generator
def gen_latent_space(latent_dims, n_samples):

    ans = None
    for i in range(n_samples):

        # noise = np.random.normal(0.0, 1.0, size=latent_shape)
        # noise = np.random.normal(0.0, 1.0, size=latent_dims)
        # noise = np.random.normal(0.5, 0.25, size=latent_shape)
        # noise = np.random.uniform(low=0.0, high=1.0, size=latent_shape)
        # noise = np.random.randn(latent_shape[0], latent_shape[1])
        noise = np.random.randn(latent_dims)
        if ans is None:
            txt = np.expand_dims(noise, axis=0)
            ans = txt
        else:
            txt = np.expand_dims(noise, axis=0)
            ans = np.concatenate((ans, txt), axis=0)
    return ans


# tfunction to smooth the fake positives
def smooth_positives(y):
	return y - 0.3 + (np.random.random(y.shape)*0.5)


# function to smooth the fake negatives
def smooth_negatives(y):
	return y + np.random.random(y.shape)*0.3


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


# generate fake true categories for the generator
def gen_fake_posclass(cat_shape, batch_size):

    sz = (batch_size, cat_shape[0])
    ans = np.ones(sz)
    # smoothing fakes
    ans = smooth_positives(ans)
    ans = ans.astype("float32")
    return ans


# generate fake negative category to train the GAN
def gen_fake_negclass(cat_shape, batch_size):

    sz = (batch_size, cat_shape[0])
    ans = np.zeros(sz)
    ans = smooth_negatives(ans)
    ans = ans.astype("float32")
    return ans


# function to generate fake labels to train the GAN
def gen_fake_labels(label_shape, batch_size):

    sz = (batch_size, label_shape[0])
    ans = np.random.randint(0,1, size=sz)
    ans = smooth_labels(ans)
    ans = ans.astype("float32")
    return ans


# function to create text similar to the original one with 5% of noise
def syth_text(data, nptc=0.05):

    ans = None
    noise = np.random.normal(0, nptc, data.shape)
    ans = data + noise
    return ans


# synthetizing a noisy std image from real data
def syth_std_img(data):

    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True, rotation_range=10)
    # datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, vertical_flip=True)
    ans = datagen.flow(samples, batch_size=1)
    ans = ans[0].astype("float32")
    ans = np.squeeze(ans, 0)
    return ans


# function to create new categories with some noise, default 5%
def syth_categories(data, nptc=0.05):

    ans = None
    noise = np.random.normal(0, nptc, data.shape)
    ans = data + noise
    return ans


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


# the function takes the ideas array, shape and configuration to render them into human understandable lenguage
# it select n number of ideas and plot them, for images, for text and for both
def plot_ideas(ideas, train_cfg, test_cfg):

    # get the index of random ideas in the set
    ideas_size = test_cfg.get("batch_size")
    gen_samples = test_cfg.get("gen_sample_size")
    data_cols = train_cfg.get("data_cols")

    # choosing non repeated ideas in the set
    rand_index = np.random.choice(ideas_size, size=gen_samples*gen_samples, replace=False)
    # print(rand_index)
    # print("ojo!!!!", len(ideas))
    # if the ideas are images or text
    if len(ideas) == 1:
        # print("data_cols:", data_cols)
        data = ideas[0]
        current_shape = data[0].shape
        # print("idea current_shape:", ideas.shape)
        # print("idea current_shape:", current_shape)

        if current_shape == train_cfg.get("img_shape"):
            render_painting(data, rand_index, train_cfg, test_cfg)

        elif current_shape == train_cfg.get("txt_shape"):
            render_wordcloud(data, rand_index, train_cfg, test_cfg)

    # if the ideas are images + text
    elif len(ideas) == 2:
        data_img = ideas[0]
        data_txt = ideas[1]
        render_painting(data_img, rand_index, train_cfg, test_cfg)
        render_wordcloud(data_txt, rand_index, train_cfg, test_cfg)


# this function takes the selected ideas and transform them into pytlot objects
def render_painting(ideas, rand_index, train_cfg, test_cfg):

    # get important data for iterating
    n_sample = test_cfg.get("gen_sample_size")
    report_fp_name = test_cfg.get("report_fn_path")
    epoch = test_cfg.get("current_epoch")

    # prep the figure
    fig, ax = plt.subplots(n_sample, n_sample, figsize=(20,20))
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
    fpn = os.path.join(report_fp_name, "img", plot_name)
    plt.savefig(fpn)
    plt.close()


# this function takes the selected ideas and translate them into pytplot objects
def render_wordcloud(ideas, rand_index, train_cfg, test_cfg):

    # get important data for iterating
    n_sample = test_cfg.get("n_samples")
    lexicon = train_cfg.get("bow_lexicon")
    tfidf_tokens = train_cfg.get("tfidf_lexicon")
    # get important data for iterating
    n_sample = test_cfg.get("gen_sample_size")
    report_fp_name = test_cfg.get("report_fn_path")
    epoch = test_cfg.get("current_epoch")
    default = {"without":1, "words":1, "or":1, "meaning":1}

    # prep the figure
    fig, ax = plt.subplots(n_sample,n_sample, figsize=(20,20))
    fig.patch.set_facecolor("xkcd:white")

    # plot images
    for i in range(n_sample*n_sample):
        # define subplot
        plt.subplot(n_sample, n_sample, 1+i)
        
        # getting the images from sample
        rand_i = rand_index[i]
        gtxt = ideas[rand_i]
        gtxt = translate_from_lexicon(gtxt, tfidf_tokens, lexicon)

        wordcloud = WordCloud(max_font_size=100,
                                min_font_size=10,
                                max_words=100,
                                min_word_length=1,
                                relative_scaling = 0.5,
                                width=600, height=400,
                                background_color="white",
                                random_state=42)
        if len(gtxt) == 0:
            gtxt = default
        
        wordcloud.generate_from_frequencies(frequencies=gtxt)
        # plt.figure()

        # turn off axis
        plt.axis("off")
        plt.imshow(wordcloud, interpolation="bilinear") #, interpolation="nearest")

    # plot leyend
    fig.suptitle("GENERATED WORDS", fontsize=50)
    fig.legend()

    # save plot to file
    plot_name = "GAN-Gen-txt-epoch%03d" % int(epoch)
    plot_name = plot_name + ".png"
    fpn = os.path.join(report_fp_name, "txt", plot_name)
    plt.savefig(fpn)
    plt.close()


# this function loads the model known lexicon into the a dictionary for the world cloud to translate
def load_lexicon(lexicon_fp):

    lexicon = gensim.corpora.Dictionary.load(lexicon_fp)
    return lexicon


# this function takes the idtf dense word vector representacion and translate it to human lenguage using the kown lexicon
def translate_from_lexicon(tfidf_corpus, tfidf_dict, lexicon):

    wordcloud = dict()

    bow_corpus = tfidf2bow(tfidf_corpus, tfidf_dict)
    wordcloud = bow2words(bow_corpus, lexicon)
    return wordcloud


# translate from tfidf token representation to bow representation
def tfidf2bow(tfidf_corpus, tfidf_dict):

    bows = dict()
    tfidf_corpus = np.asarray(tfidf_corpus, dtype="float32")
    # print(type(tfidf_corpus))

    for tfidf_doc in tfidf_corpus:

        for tfidf_token in tfidf_doc:

            bows = get_similars(tfidf_token, bows, tfidf_dict)

    return bows


# stablish if the tfidf representation of a token is similar to the one in the tfidf dictionar
def get_similars(tfidf_token, bows, tfidf_dict):

    ans = bows
    ans = isclose_in(tfidf_token, bows, tfidf_dict)
    return ans


# this function return the similar values of the tfidf value with a token id and a count
def isclose_in(token, token_dict, cmp_tokens, tol=0.0001):

    for tcmp in cmp_tokens:
        for key, value in tcmp.items():

            if math.isclose(token, value, rel_tol=tol) and (key not in token_dict.keys()):
                token_dict.update({key:1})
            
            elif math.isclose(token, value, rel_tol=tol) and (key in token_dict.keys()):
                count = token_dict[key]
                count = count + 1
                token_dict.update({key:count})
    return token_dict


def bow2words(bow_txt, lexicon):

    words = dict()

    for key, value in bow_txt.items():
        token = lexicon.get(key)
        td = {token:value}
        # word = id2token.get(key)
        # td = {word:value}
        words.update(td)
    return words


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
    fpn = os.path.join(report_fp_name, "img", plot_name)
    plt.savefig(fpn)
    plt.close()


# create a line plot of loss for the gan and save to file
def plot_metrics(disr_hist, disf_hist, gan_hist, report_fp_name, epoch):

    # print(len(disr_hist[0]), len(disf_hist[0]), len(gan_hist[0]))
    if len(gan_hist[0]) == 2:
        plot_simple_metrics(disr_hist, disf_hist, gan_hist, report_fp_name, epoch)

    elif len(gan_hist[0]) == 5:
        plot_multi_metrics(disr_hist, disf_hist, gan_hist, report_fp_name, epoch)


# create a line plot of loss for the gan and save to file
def plot_simple_metrics(disr_hist, disf_hist, gan_hist, report_fp_name, epoch):

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
    ax1.set_title("Accuracy")
    ax2.set_title("Loss")
    ax1.set(xlabel = "Epoch [cycle]", ylabel = "Acc")
    ax2.set(xlabel = "Epoch [cycle]", ylabel = "Loss")
    fig.legend()

    # save plot to file
    plot_name = "GAN-learn-curve-epoch%03d" % int(epoch)
    plot_name = plot_name + ".png"
    fpn = os.path.join(report_fp_name, "learn", plot_name)
    plt.savefig(fpn)
    plt.close()


def plot_multi_metrics(disr_hist, disf_hist, gan_hist, report_fp_name, epoch):

    # reporting results
    disr_hist = np.array(disr_hist)
    disf_hist = np.array(disf_hist)
    gan_hist = np.array(gan_hist)

    fig, ax  = plt.subplots(2,2, figsize=(18,9))
    # img loss, img acc, txt loss, txt acc
    ax1 = ax[0][0]
    ax2 = ax[0][1]
    ax3 = ax[1][0]
    ax4 = ax[1][1]
    fig.patch.set_facecolor("xkcd:white")

    # img loss
    ax1.plot(disr_hist[:,1], "royalblue", label="Loss-img: R-Dis")
    ax1.plot(disf_hist[:,1], "crimson", label="Loss-img: F-Dis")
    ax1.plot(gan_hist[:,1], "blueviolet", label="Loss-img: GAN/Gen")

    # txt loss
    ax2.plot(disr_hist[:,2], "goldenrod", label="Loss-txt: R-Dis")
    ax2.plot(disf_hist[:,2], "firebrick", label="Loss-txt: F-Dis")
    ax2.plot(gan_hist[:,2], "forestgreen", label="Loss-txt: GAN/Gen")

    # plot leyend
    fig.suptitle("IMG & TXT LEARNING BEHAVIOR", fontsize=20)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_title("Images")
    ax2.set_title("Text")
    # ax1.set(xlabel = "Epoch [cycle]", ylabel = "Loss")
    ax1.set(xlabel = "Epoch [cycle]", ylabel = "Loss")
    ax2.set(xlabel = "Epoch [cycle]", ylabel = "Loss")
    # fig.legend()

    # img acc
    ax3.plot(disr_hist[:,3], "royalblue", label="Acc-img: R-Dis")
    ax3.plot(disf_hist[:,3], "crimson", label="Acc-img: F-Dis")
    ax3.plot(gan_hist[:,3], "blueviolet", label="Acc-img: GAN/Gen")

    # txt acc
    ax4.plot(disr_hist[:,4], "goldenrod", label="Acc-txt: R-Dis")
    ax4.plot(disf_hist[:,4], "firebrick", label="Acc-txt: F-Dis")
    ax4.plot(gan_hist[:,4], "forestgreen", label="Acc-txt: GAN/Gen")

    # plot leyend
    ax3.grid(True)
    ax4.grid(True)
    # ax3.set_title("Txt Loss")
    # ax4.set_title("Txt Accuracy")
    ax3.set(xlabel = "Epoch [cycle]", ylabel = "Acc")
    ax4.set(xlabel = "Epoch [cycle]", ylabel = "Acc")
    fig.legend()

    # save plot to file
    plot_name = "GAN-learn-curve-epoch%03d" % int(epoch)
    plot_name = plot_name + ".png"
    fpn = os.path.join(report_fp_name, "learn", plot_name)
    plt.savefig(fpn)
    plt.close()


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


# function to calculate the loss and accuracy avg in multiple batchs of an epoch V2 with numpy
def epoch_avg_metrics(log):

    np_logs = np.array(log, dtype="float")
    avg_log = np.mean(np_logs, axis=0)
    ans = avg_log.tolist()
    return ans


# function to save model, needs the dirpath, the name and the datetime to save
def export_model(model, models_fp_name, filename, datetime):

    ss = True
    sln = True
    fext = "png"
    fpn = filename + "-" + datetime
    fpn = filename + "." + fext
    fpn = os.path.join(models_fp_name, fpn)
    plot_model(model, to_file=fpn, show_shapes=ss, show_layer_names=sln)


# function to format data to save in file
def format_metrics(disr_history, disf_history, gan_history):

    headers, data = None, None

    disr_hist = np.array(disr_history)
    disf_hist = np.array(disf_history)
    gan_hist = np.array(gan_history)

    # formating file headers
    if gan_hist.shape[1] == 2:
        headers = ["dis_loss_real", 
                    "dis_acc_real", 
                    "dis_loss_fake", 
                    "dis_acc_fake", 
                    "gen_gan_loss", 
                    "gen_gan_acc"]

    if gan_hist.shape[1] == 5:
        headers = ["dis_loss_real", 
                    "dis_loss_real_img",
                    "dis_loss_real_txt",
                    "dis_acc_real_img", 
                    "dis_acc_real_txt",
                    "dis_loss_fake", 
                    "dis_loss_fake_img",
                    "dis_loss_fake_txt",
                    "dis_acc_fake_img", 
                    "dis_acc_fake_txt",
                    "gen_loss_fake", 
                    "gen_loss_fake_img",
                    "gen_loss_fake_txt",
                    "gen_acc_fake_img", 
                    "gen_acc_fake_txt",]

    # adding all formatted data into list
    data = np.concatenate((disr_hist, disf_hist, gan_hist), axis=1)

    return data, headers

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

# function to write data in csv file
def read_metrics(report_fn_path, filename):

    # print(report_fn_path, filename)
    fpn = filename + "-train-history.csv"
    fpn = os.path.join(report_fn_path, fpn)
    tdata = pd.read_csv(
                        fpn,
                        sep=",",
                        encoding="utf-8",
                        engine="python",
                        quoting=csv.QUOTE_ALL
                        )
    return tdata

def extract_history(data):

    disr_history, disf_history, gan_history = None, None, None

    # formating file headers
    # print("columns len():", len(data.columns))
    # print("columns:\n", list(data.columns.values))
    if len(data.columns) == 2*3:
        disr_history = data[["dis_loss_real",
                            "dis_acc_real",]]
        disf_history = data[["dis_loss_fake",
                             "dis_acc_fake",]]
        gan_history = data[["gen_gan_loss",
                            "gen_gan_acc",]]

    if len(data.columns) == 5*3:

        disr_history = data[["dis_loss_real",
                             "dis_loss_real_img",
                             "dis_loss_real_txt",
                             "dis_acc_real_img",
                             "dis_acc_real_txt",]]
        disf_history = data[["dis_loss_fake",
                             "dis_loss_fake_img",
                             "dis_loss_fake_txt",
                             "dis_acc_fake_img",
                             "dis_acc_fake_txt",]]
        gan_history = data[["gen_loss_fake",
                            "gen_loss_fake_img",
                            "gen_loss_fake_txt",
                            "gen_acc_fake_img",
                            "gen_acc_fake_txt",]]

    disr_history = disr_history.values.tolist()
    disf_history = disf_history.values.tolist()
    gan_history = gan_history.values.tolist()
    # print("checking stuff!!!")
    # print("gan_history:\n", type(gan_history), len(gan_history))
    # print(gan_history[0:5])
    return disr_history, disf_history, gan_history

# function to load the matrics form the gan/gen/dis history in csv
def load_metrics(report_fn_path, filename):

    data = read_metrics(report_fn_path, filename)
    disr_history, disf_history, gan_history = extract_history(data)
    return disr_history, disf_history, gan_history


# function to safe the loss/acc logs in training for the gan/gen/dis models
def save_metrics(disr_history, disf_history, gan_history, report_fn_path, filename):

    data, headers = format_metrics(disr_history, disf_history, gan_history)
    write_metrics(data, headers, report_fn_path, filename)


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


# function to test the model while training
def test_model(gen_model, dis_model, data, data_shape, train_cfg, test_cfg): 

    dataset_size = test_cfg.get("dataset_size")
    batch_size = test_cfg.get("batch_size")
    synth_batch = test_cfg.get("synth_batch")
    epoch = int(test_cfg.get("current_epoch"))
    report_fn_path = test_cfg.get("report_fn_path")
    gen_samples = test_cfg.get("gen_sample_size") 
    balance_batch = test_cfg.get("balance_batch")
    split_batch = int(batch_size/2)

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

    # plotting gen ideas
    # ideas = (fake_data[0], fake_data[1])
    # plot_ideas(ideas, train_cfg, test_cfg)

    # test metrics
    test_real, test_fake = None, None
    # summarize discriminator performance
    print("Batch Size %d -> Samples: Fake: %d & Real: %d" % (batch_size*synth_batch, split_batch, split_batch))

    # 1 output, img or txt
    if len(data) == 2:
        ideas = (fake_data[0],)
        print(len(ideas))
        plot_ideas(ideas, train_cfg, test_cfg)
        test_real, test_fake = test_gan(dis_model, real_data, fake_data, batch_size)
        print(">>> Test Fake -> Acc: %.3f || Loss: %.3f" % (test_fake[1], test_fake[0]))
        print(">>> Test Real -> Acc: %.3f || Loss: %.3f" % (test_real[1], test_real[0]))

    # 2 output, img + txt and labels conditioned
    elif len(data) == 3 and data_shape.get("conditioned") == True:
        ideas = (fake_data[0],)
        plot_ideas(ideas, train_cfg, test_cfg)
        test_real, test_fake = test_cgan(dis_model, real_data, fake_data, batch_size)
        print(">>> Test Fake -> Acc: %.3f || Loss: %.3f" % (test_fake[1], test_fake[0]))
        print(">>> Test Real -> Acc: %.3f || Loss: %.3f" % (test_real[1], test_real[0]))

    # 2 output, img + txt unconditioned
    elif len(data) == 3 and data_shape.get("conditioned") == False:
        ideas = (fake_data[0], fake_data[1])
        plot_ideas(ideas, train_cfg, test_cfg)
        test_real, test_fake = test_multi_gan(dis_model, real_data, fake_data, batch_size)
        print(">>> Test Fake -> Acc: %.3f || Loss: %.3f" % (test_fake[1], test_fake[0]))
        print(">>> Test Real -> Acc: %.3f || Loss: %.3f" % (test_real[1], test_real[0]))

    # 2 outputs, img + txt and label conditioned
    elif len(data) == 4:
        ideas = (fake_data[0], fake_data[1])
        plot_ideas(ideas, train_cfg, test_cfg)
        test_real, test_fake = test_multi_cgan(dis_model, real_data, fake_data, batch_size)
        # print("test_real, test_fake\n", test_real, "\n", test_fake)

        # general learning metrics
        log_msg = "GENERAL GAN METRICS:\n Discriminator -> [R-Loss: %.3f, F-Loss: %.3f]\n" % (test_real[0], test_real[0])
        # img learning metrics
        log_msg = "%s SYNTH-IMG Metrics:\n" % log_msg
        log_msg = "%s Discriminator -> Real=[loss: %.3f, acc: %.3f];" % (log_msg, test_real[1], test_real[3])
        log_msg = "%s Fake=[loss: %.3f, acc: %.3f]\n" % (log_msg, test_fake[1], test_fake[3])
        # txt learning metrics
        log_msg = "%s SYNTH-TXT Metrics:\n" % log_msg
        log_msg = "%s Discriminator -> Real=[loss: %.3f, acc: %.3f];" % (log_msg, test_real[2], test_real[4])
        log_msg = "%s Fake=[loss: %.3f, acc: %.3f]" % (log_msg, test_fake[2], test_fake[4])
        print(log_msg)


# special function to train the GAN
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
# def train(gen_model, dis_model, gan_model, X_img, X_txt, y, labels, epochs, batch_size, save_intervas, fn_config):
def training_model_old(gen_model, dis_model, gan_model, data, train_cfg): # epochs, batch_size, save_intervas, fn_config

    # sample size
    dataset_size = train_cfg.get("dataset_size")

    # data shape for the generator
    data_shape = {
        "latent_dims": train_cfg.get("latent_dims"),
        "cat_shape": train_cfg.get("cat_shape"),
        "txt_shape": train_cfg.get("txt_shape"),
        "label_shape": train_cfg.get("label_shape"),
        "conditioned": train_cfg.get("conditioned"),
        "data_cols": train_cfg.get("data_cols"),
        }
    # print(data_shape)

    # augmentation factor
    synth_batch = train_cfg.get("synth_batch")
    balance_batch = train_cfg.get("balance_batch")
    n = train_cfg.get("gen_sample_size")
    epochs = train_cfg.get("max_epochs")
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
    learning_history = train_cfg.get("learning_history")

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
        "current_epoch": None,
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
            real_data, fake_data = drift_labels(real_data, fake_data, half_batch, 0.15)

            # TODO transfor this in 1 function train_model()...
            dhf, dhr, gh = None, None, None

            if len(data) == 2:
                dhr, dhf, gh = train_gan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 3 and data_shape.get("conditioned") == True:
                dhr, dhf, gh = train_cgan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 3 and data_shape.get("conditioned") == False:
                # TODO need to implement this function!!!
                dhr, dhf, gh = train_multi_gan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 4:
                dhr, dhf, gh = train_multi_cgan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            # epoch log
            # print("Epoch opti_loss_acc:\ndhr, dhf, g\n", dhr, dhf, gh)
            ep_disr_hist.append(dhr)
            ep_disf_hist.append(dhf)
            ep_gan_hist.append(gh)

            # printing in console learning metrics with single output
            if len(data) in (2,3):
                log_msg = ">>> Epoch: %d, B/Ep: %d/%d, Batch S: %d\n" %(ep+1, batch+1, batch_per_epoch, batch_size*synth_batch)
                log_msg = "%s -> [R-Dis loss: %.3f, acc: %.3f]" % (log_msg, dhr[0], dhr[1])
                log_msg = "%s || [F-Dis loss: %.3f, acc: %.3f]" % (log_msg, dhf[0], dhf[1])
                log_msg = "%s || [Gen loss: %.3f, acc: %.3f]" % (log_msg, gh[0], gh[1])
                print(log_msg)

            # printing in console learning metrics with multiple outputs
            elif len(data) == 4:
                # training stage
                log_msg = ">>> Epoch: %d, B/Ep: %d/%d, Batch S: %d\n" %(ep+1, batch+1, batch_per_epoch, batch_size*synth_batch)

                # general learning metrics
                log_msg = "%s GENERAL GAN METRICS:\n Discriminator -> [R-Loss: %.3f, F-Loss: %.3f]" % (log_msg, dhr[0], dhf[0])
                log_msg = "%s || Generator -> [Loss: %.3f]\n" % (log_msg, gh[0])
                # img learning metrics
                log_msg = "%s SYNTH-IMG Metrics:\n" % log_msg
                log_msg = "%s Discriminator -> Real=[loss: %.3f, acc: %.3f];" % (log_msg, dhr[1], dhr[3])
                log_msg = "%s Fake=[loss: %.3f, acc: %.3f]" % (log_msg, dhf[1], dhf[3])
                log_msg = "%s || Generator -> img=[loss: %.3f, acc: %.3f]\n" % (log_msg, gh[1], gh[3])
                # txt learning metrics
                log_msg = "%s SYNTH-TXT Metrics:\n" % log_msg
                log_msg = "%s Discriminator -> Real=[loss: %.3f, acc: %.3f];" % (log_msg, dhr[2], dhr[4])
                log_msg = "%s Fake=[loss: %.3f, acc: %.3f]" % (log_msg, dhf[2], dhf[4])
                log_msg = "%s || Generator -> txt=[loss: %.3f, acc: %.3f]" % (log_msg, gh[2], gh[4])
                print(log_msg)

        # updating epoch
        test_cfg["current_epoch"] = ep
        # record history for epoch
        disr_hist.append(epoch_avg_metrics(ep_disr_hist))
        disf_hist.append(epoch_avg_metrics(ep_disf_hist))
        gan_hist.append(epoch_avg_metrics(ep_gan_hist))

		# evaluate the model performance sometimes
        if (ep) % check_intervas == 0:
            print("Epoch:", ep+1, "Testing model training process...")
            
            # test_model(gen_model, dis_model, data, data_shape, test_cfg) #, synth_batch)
            test_model(gen_model, dis_model, data, data_shape, train_cfg, test_cfg)
            print("Ploting results...")
            plot_metrics(disr_hist, disf_hist, gan_hist, report_fn_path, ep)
            print("Saving metrics...")
            save_metrics(disr_hist, disf_hist, gan_hist, report_fn_path, gan_model_name)

		# saving the model sometimes
        if (ep) % save_intervas == 0:
            print("Epoch:", ep+1, "Saving the training progress in model...")
            save_models(dis_model, gen_model, gan_model, train_cfg, test_cfg)
            print("Cleaning old models...")
            clear_models(train_cfg, test_cfg)
        
        train_time = lapse_time(train_time, ep)

# function to continue training from history
def training_model(gen_model, dis_model, gan_model, data, train_cfg):

    # sample size
    dataset_size=train_cfg.get("dataset_size")

    # data shape for the generator
    data_shape={
        "latent_dims": train_cfg.get("latent_dims"),
        "cat_shape": train_cfg.get("cat_shape"),
        "txt_shape": train_cfg.get("txt_shape"),
        "label_shape": train_cfg.get("label_shape"),
        "conditioned": train_cfg.get("conditioned"),
        "data_cols": train_cfg.get("data_cols"),
        }
    # print(data_shape)

    # augmentation factor
    synth_batch = train_cfg.get("synth_batch")
    balance_batch = train_cfg.get("balance_batch")
    trained = train_cfg.get("trained")
    trained_epochs = train_cfg.get("trained_epochs")
    epochs = train_cfg.get("max_epochs")

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
    learning_history = train_cfg.get("learning_history")

	# prepare lists for storing stats each epoch
    disr_hist, disf_hist, gan_hist = list(), list(), list()

    train_time = None

    # train dict config
    test_cfg = {
        "report_fn_path": report_fn_path,
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "synth_batch": synth_batch,
        "gen_sample_size": train_cfg.get("gen_sample_size"),
        "current_epoch": trained_epochs,
    }
    # print(test_cfg)
    # print("trained?", trained)

    if trained == True:
        print("models are already trained!...")
        print("Loading models...")        
        load_models(dis_model, gen_model, gan_model, train_cfg, test_cfg)
        print("Loading metrics...")
        disr_hist, disf_hist, gan_hist = load_metrics(report_fn_path, 
                                                        gan_model_name)

    ep = trained_epochs
    # iterating in training epochs:
    while ep < epochs+1:
    # for ep in range(epochs+1):
        # epoch logs
        ep_disf_hist, ep_disr_hist, ep_gan_hist = list(), list(), list()
        train_time = datetime.datetime.now()

        # iterating over training batchs
        for batch in range(batch_per_epoch):
            # pass

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
            real_data, fake_data = drift_labels(real_data, fake_data, half_batch, 0.15)

            # TODO transfor this in 1 function train_model()...
            dhf, dhr, gh = None, None, None

            if len(data) == 2:
                dhr, dhf, gh = train_gan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 3 and data_shape.get("conditioned") == True:
                dhr, dhf, gh = train_cgan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 3 and data_shape.get("conditioned") == False:
                # TODO need to implement this function!!!
                dhr, dhf, gh = train_multi_gan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            elif len(data) == 4:
                dhr, dhf, gh = train_multi_cgan(dis_model, gan_model, real_data, fake_data, batch_size, data_shape)

            # epoch log
            # print("Epoch opti_loss_acc:\ndhr, dhf, g\n", dhr, dhf, gh)
            ep_disr_hist.append(dhr)
            ep_disf_hist.append(dhf)
            ep_gan_hist.append(gh)

            # printing in console learning metrics with single output
            if len(data) in (2,3):
                log_msg = ">>> Epoch: %d, B/Ep: %d/%d, Batch S: %d\n" %(ep+1, batch+1, batch_per_epoch, batch_size*synth_batch)
                log_msg = "%s -> [R-Dis loss: %.3f, acc: %.3f]" % (log_msg, dhr[0], dhr[1])
                log_msg = "%s || [F-Dis loss: %.3f, acc: %.3f]" % (log_msg, dhf[0], dhf[1])
                log_msg = "%s || [Gen loss: %.3f, acc: %.3f]" % (log_msg, gh[0], gh[1])
                print(log_msg)

            # printing in console learning metrics with multiple outputs
            elif len(data) == 4:
                # training stage
                log_msg = ">>> Epoch: %d, B/Ep: %d/%d, Batch S: %d\n" %(ep+1, batch+1, batch_per_epoch, batch_size*synth_batch)

                # general learning metrics
                log_msg = "%s GENERAL GAN METRICS:\n Discriminator -> [R-Loss: %.3f, F-Loss: %.3f]" % (log_msg, dhr[0], dhf[0])
                log_msg = "%s || Generator -> [Loss: %.3f]\n" % (log_msg, gh[0])
                # img learning metrics
                log_msg = "%s SYNTH-IMG Metrics:\n" % log_msg
                log_msg = "%s Discriminator -> Real=[loss: %.3f, acc: %.3f];" % (log_msg, dhr[1], dhr[3])
                log_msg = "%s Fake=[loss: %.3f, acc: %.3f]" % (log_msg, dhf[1], dhf[3])
                log_msg = "%s || Generator -> img=[loss: %.3f, acc: %.3f]\n" % (log_msg, gh[1], gh[3])
                # txt learning metrics
                log_msg = "%s SYNTH-TXT Metrics:\n" % log_msg
                log_msg = "%s Discriminator -> Real=[loss: %.3f, acc: %.3f];" % (log_msg, dhr[2], dhr[4])
                log_msg = "%s Fake=[loss: %.3f, acc: %.3f]" % (log_msg, dhf[2], dhf[4])
                log_msg = "%s || Generator -> txt=[loss: %.3f, acc: %.3f]" % (log_msg, gh[2], gh[4])
                print(log_msg)

        # updating epoch
        test_cfg["current_epoch"] = ep
        # record history for epoch
        disr_hist.append(epoch_avg_metrics(ep_disr_hist))
        disf_hist.append(epoch_avg_metrics(ep_disf_hist))
        gan_hist.append(epoch_avg_metrics(ep_gan_hist))

		# evaluate the model performance sometimes
        if (ep) % check_intervas == 0:
            print("Epoch:", ep+1, "Testing model training process...")
            test_model(gen_model, dis_model, data, data_shape, train_cfg, test_cfg)
            print("Ploting results...")
            plot_metrics(disr_hist, disf_hist, gan_hist, report_fn_path, ep)
            print("Saving metrics...")
            save_metrics(disr_hist, disf_hist, gan_hist, report_fn_path, gan_model_name)

		# saving the model sometimes
        if (ep) % save_intervas == 0:
            print("Epoch:", ep+1, "Saving the training progress in model...")
            save_models(dis_model, gen_model, gan_model, train_cfg, test_cfg)
            print("Cleaning old models...")
            clear_models(train_cfg, test_cfg)
        
        train_time = lapse_time(train_time, ep)
        ep = ep + 1

# function to save GAN models
def save_models(dis_model, gen_model, gan_model, train_cfg, test_cfg):

    ep = test_cfg.get("current_epoch") # = ep
    epoch_sufix = "-epoch%d" % int(ep)

    model_fn_path = train_cfg.get("models_fn_path")
    dis_model_name = train_cfg.get("dis_model_name")
    gen_model_name = train_cfg.get("gen_model_name")
    gan_model_name = train_cfg.get("gan_model_name")

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


# function to save GAN models
def load_models(dis_model, gen_model, gan_model, train_cfg, test_cfg):

    ep = test_cfg.get("current_epoch")  # = ep
    # print(ep)
    epoch_sufix = "-epoch%d" % int(ep)

    model_fn_path = train_cfg.get("models_fn_path")
    dis_model_name = train_cfg.get("dis_model_name")
    gen_model_name = train_cfg.get("gen_model_name")
    gan_model_name = train_cfg.get("gan_model_name")

    # epoch_sufix = "-last"
    epoch_sufix = str(epoch_sufix)
    dis_mn = dis_model_name + epoch_sufix
    gen_mn = gen_model_name + epoch_sufix
    gan_mn = gan_model_name + epoch_sufix

    dis_path = os.path.join(model_fn_path, "Dis")
    gen_path = os.path.join(model_fn_path, "Gen")
    gan_path = os.path.join(model_fn_path, "GAN")

    load_model(dis_model, dis_path, dis_mn)
    load_model(gen_model, gen_path, gen_mn)
    load_model(gan_model, gan_path, gan_mn)
    # return dis_model, gen_model, gan_model


def clear_models(train_cfg, test_cfg):

    # epoch_sufix = "-epoch%d" % int(ep)
    model_fn_path = train_cfg.get("models_fn_path")
    dis_path = os.path.join(model_fn_path, "Dis")
    gen_path = os.path.join(model_fn_path, "Gen")
    gan_path = os.path.join(model_fn_path, "GAN")
    max_files = train_cfg.get("max_save_models")

    list_path = (dis_path, gen_path, gan_path)
    rmv_path = list()

    for path in list_path:

        files = os.listdir(path)
        filepaths = list()

        for f in files:
            fp = os.path.join(path, f)
            filepaths.append(fp)

        # print(filepaths)
        filepaths.sort(key=os.path.getctime, reverse=True)
        # print("files!!!!", filepaths)

        if len(filepaths) > max_files:

            for del_file in filepaths[max_files:]:
                os.remove(del_file)


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

    return dhr, dhf, gh


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

    return dhr, dhf, gh


def train_multi_gan(dis_model, gan_model, real_data, fake_data, batch_size, dataset_shape):
    pass


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

    rd = False

    # train for real samples batch
    dhr = dis_model.train_on_batch([Xi_real, Xt_real, Xl_real], [y_real, y_real], return_dict=rd)
    # train for fake samples batch
    dhf = dis_model.train_on_batch([Xi_fake, Xt_fake, Xl_fake], [y_fake, y_fake], return_dict=rd)

    # prepare text and inverted categories from the latent space as input for the generator
    latent_gen, yl_gen, y_gen = gen_latent_data(dataset_shape, batch_size)

    # update the generator via the discriminator's error
    gh = gan_model.train_on_batch([latent_gen, yl_gen], [y_gen, y_gen], return_dict=rd)

    return dhr, dhf, gh


def test_gan(dis_model, real_data, fake_data, batch_size):
    
    # drift labels to confuse the discriminator
    real_data, fake_data = drift_labels(real_data, fake_data, batch_size, 0.15)

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


def test_cgan(dis_model, real_data, fake_data, batch_size):
    
    # drift labels to confuse the model
    real_data, fake_data = drift_labels(real_data, fake_data, batch_size, 0.15)

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


def test_multi_gan(dis_model, real_data, fake_data, batch_size):
    pass


def test_multi_cgan(dis_model, real_data, fake_data, batch_size):

    # drift labels to confuse the model
    real_data, fake_data = drift_labels(real_data, fake_data, batch_size, 0.15)

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

    rd = False

    # evaluate model
    test_real = dis_model.evaluate([Xi_real, Xt_real, Xl_real], [y_real, y_real], verbose=0, return_dict=rd)
    test_fake = dis_model.evaluate([Xi_fake, Xt_fake, Xl_fake], [y_fake, y_fake], verbose=0, return_dict=rd)
    # print("test_real, test_fake\n", test_real, test_fake)

    return test_real, test_fake

# # ML Model Definition
#
# ## Image GAN

# convolutional generator for images


def create_img_generator(latent_dims, model_cfg):

    # MODEL CONFIG
    # def of the latent space size for the input
    gen_model_name = model_cfg.get("gen_model_name")
    latent_features = model_cfg.get("latent_features")
    latent_filters = model_cfg.get("latent_filters")
    latent_dense = latent_features*latent_features*latent_filters
    # latent_input = latent_shape[0]*latent_shape[1]
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

    # kernet initialization config
    initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    batchep = 0.00001

    # LAYER CREATION
    # input layer
    in_latent = Input(shape=latent_dims, name="ImgGenIn")

    # dense layer
    lyr1 = Dense(latent_dense,
                 activation=hid_lyr_act,
                 name="ImgGenDense_1")(in_latent)

    # reshape layer 1D-> 2D (rbg image)
    lyr2 = Reshape(latent_img_shape, name="ImgGenReshape_2")(lyr1)

    # transpose conv2D layer
    lyr3 = Conv2DTranspose(int(filters), kernel_size=ksize,
                           kernel_initializer=initializer,
                           strides=stsize, activation=hid_lyr_act,
                           padding=pad, name="ImgGenConv2D_3")(lyr2)

    # batch normalization + drop layers to avoid overfit
    lyr4 = BatchNormalization(name="ImgGenBN_4",
                              epsilon=batchep)(lyr3)
    lyr5 = Dropout(hid_ldrop, name="ImgGenDrop_5")(lyr4)

    # transpose conv2D layer
    lyr6 = Conv2DTranspose(int(filters/2), kernel_size=ksize,
                           kernel_initializer=initializer,
                           strides=stsize, activation=hid_lyr_act,
                           padding=pad, name="ImgGenConv2D_6")(lyr5)

    # batch normalization + drop layers to avoid overfit
    lyr7 = BatchNormalization(name="ImgGenBN_7",
                              epsilon=batchep)(lyr6)
    lyr8 = Dropout(hid_ldrop, name="ImgGenDrop_8")(lyr7)

    # transpose conv2D layer
    lyr9 = Conv2DTranspose(int(filters/4), kernel_size=ksize,
                           kernel_initializer=initializer,
                           strides=stsize, activation=out_lyr_act,
                           padding=pad, name="ImgGenConv2D_9")(lyr8)

    # batch normalization + drop layers to avoid overfit
    lyr10 = BatchNormalization(name="ImgGenBN_10",
                               epsilon=batchep)(lyr9)
    lyr11 = Dropout(hid_ldrop, name="ImgGenDrop_11")(lyr10)

    # transpose conv2D layer
    lyr12 = Conv2DTranspose(int(filters/8), kernel_size=ksize,
                            kernel_initializer=initializer,
                            strides=stsize, activation=hid_lyr_act,
                            padding=pad, name="ImgGenConv2D_12")(lyr11)

    # batch normalization + drop layers to avoid overfit
    lyr13 = BatchNormalization(name="ImgGenBN_13",
                               epsilon=batchep)(lyr12)
    lyr14 = Dropout(hid_ldrop, name="ImgGenDrop_14")(lyr13)

    # transpose conv2D layer
    lyr15 = Conv2DTranspose(int(filters/16), kernel_size=out_ksize,
                            kernel_initializer=initializer,
                            strides=out_stsize, activation=hid_lyr_act,
                            padding=pad, name="ImgGenConv2D_15")(lyr14)

    # batch normalization + drop layers to avoid overfit
    lyr16 = BatchNormalization(name="ImgGenBN_16",
                               epsilon=batchep)(lyr15)
    lyr17 = Dropout(hid_ldrop, name="ImgGenDrop_17")(lyr16)

    # # transpose conv2D layer
    # lyr18 = Conv2DTranspose(int(filters/32), kernel_size=ksize,
    #                         kernel_initializer=initializer,
    #                         strides=stsize, activation=hid_lyr_act,
    #                         padding=pad, name="ImgGenConv2D_18")(lyr17)

    # # batch normalization + drop layers to avoid overfit
    # lyr19 = BatchNormalization(name="ImgGenBN_19",
    #                             epsilon=batchep)(lyr18)
    # lyr20 = Dropout(hid_ldrop, name="ImgGenDrop_20")(lyr19)

    # output layer
    out_img = Conv2D(out_filters, kernel_size=out_ksize,
                     kernel_initializer=initializer,
                     strides=out_stsize, activation=out_lyr_act,
                     padding=out_pad, input_shape=img_shape,
                     name="ImgGenOut")(lyr17)  # (lyr20)

    # MODEL DEFINITION
    model = Model(inputs=in_latent, outputs=out_img, name=gen_model_name)
    return model


# convolutional discriminator for images
def create_img_discriminator(img_shape, model_cfg):

    # MODEL CONFIG
    # input layer config, image classification
    dis_model_name = model_cfg.get("dis_model_name")
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

    # kernet initialization config
    initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    batchep = 0.00001

    # LAYER CREATION
    # input layer
    in_img = Input(shape=img_shape, name="DisImgIn")

    # DISCRIMINATOR LAYERS
    # intermediate conv layer 64 filters
    lyr1 = Conv2D(int(in_filters/64), kernel_size=in_ksize,
                  kernel_initializer=initializer,
                  padding=in_pad, activation=in_lyr_act,
                  strides=in_stsize, name="ImgDisConv2D_1")(in_img)

    # batch normalization + drop layers to avoid overfit
    lyr2 = BatchNormalization(name="ImgDisBN_2",
                              epsilon=batchep)(lyr1)
    lyr3 = Dropout(hid_ldrop, name="ImgDisDrop_3")(lyr2)

    # intermediate conv layer 128 filters
    lyr4 = Conv2D(int(in_filters/32), kernel_size=ksize,
                  kernel_initializer=initializer,
                  padding=pad, activation=hid_lyr_act,
                  strides=stsize, name="ImgDisConv2D_4")(lyr3)

    # batch normalization + drop layers to avoid overfit
    lyr5 = BatchNormalization(name="ImgDisBN_5",
                              epsilon=batchep)(lyr4)
    lyr6 = Dropout(hid_ldrop, name="ImgDisDrop_6")(lyr5)

    # intermediate conv layer 256 filters
    sp_stsize = (1, 1)
    lyr7 = Conv2D(int(in_filters/16), kernel_size=ksize,
                  kernel_initializer=initializer,
                  padding=pad, activation=hid_lyr_act,
                  strides=sp_stsize, name="ImgDisConv2D_7")(lyr6)

    # batch normalization + drop layers to avoid overfit
    lyr8 = BatchNormalization(name="ImgDisBN_8",
                              epsilon=batchep)(lyr7)
    lyr9 = Dropout(hid_ldrop, name="ImgDisDrop_9")(lyr8)

    # intermediate conv layer 512 filters
    lyr10 = Conv2D(int(filters/8), kernel_size=ksize,
                   kernel_initializer=initializer,
                   padding=pad, activation=hid_lyr_act,
                   strides=stsize, name="ImgDisConv2D_10")(lyr9)

    # batch normalization + drop layers to avoid overfit
    lyr11 = BatchNormalization(name="ImgDisBN_11",
                               epsilon=batchep)(lyr10)
    lyr12 = Dropout(hid_ldrop, name="ImgDisDrop_12")(lyr11)

    # intermediate conv layer 1024 filters
    lyr13 = Conv2D(int(filters/4), kernel_size=ksize,
                   kernel_initializer=initializer,
                   padding=pad, activation=hid_lyr_act,
                   strides=stsize, name="ImgDisConv2D_13")(lyr12)

    # batch normalization + drop layers to avoid overfit
    lyr14 = BatchNormalization(name="ImgDisBN_14",
                               epsilon=batchep)(lyr13)
    lyr15 = Dropout(hid_ldrop, name="ImgDisDrop_15")(lyr14)

    # intermediate conv layer
    lyr16 = Conv2D(int(filters/2), kernel_size=ksize,
                   padding=pad, activation=hid_lyr_act,
                   strides=stsize, name="ImgDisConv2D_16")(lyr15)

    # batch normalization + drop layers to avoid overfit
    lyr17 = BatchNormalization(name="ImgDisBN_17",
                               epsilon=batchep)(lyr16)
    lyr18 = Dropout(hid_ldrop, name="ImgDisDrop_18")(lyr17)

    # # intermediate conv layer
    # lyr19 = Conv2D(int(filters), kernel_size=ksize,
    #                 padding=pad, activation=hid_lyr_act,
    #                 strides=stsize, name="ImgDisConv2D_19")(lyr18)

    # # batch normalization + drop layers to avoid overfit
    # lyr20 = BatchNormalization(name="ImgDisBN_20",
    #                             epsilon=batchep)(lyr19)
    # lyr21 = Dropout(hid_ldrop, name="ImgDisDrop_21")(lyr20)

    # flatten from 2D to 1D
    lyr22 = Flatten(name="ImgDisFlat_22")(lyr18)

    # dense classifier layers
    lyr23 = Dense(int(mid_disn), activation=hid_cls_act,
                  name="ImgDisDense_23")(lyr22)
    lyr24 = Dense(int(mid_disn/2), activation=hid_cls_act,
                  name="ImgDisDense_24")(lyr23)
    # drop layer
    lyr25 = Dropout(hid_ldrop, name="ImgDisDrop_25")(lyr24)

    # dense classifier layers
    lyr26 = Dense(int(mid_disn/4), activation=hid_cls_act,
                  name="ImgDisDense_26")(lyr25)
    lyr27 = Dense(int(mid_disn/8), activation=hid_cls_act,
                  name="ImgDisDense_27")(lyr26)
    # drop layer
    lyr28 = Dropout(hid_ldrop, name="ImgDisDrop_28")(lyr27)

    # dense classifier layers
    lyr29 = Dense(int(mid_disn/16), activation=hid_cls_act,
                  name="ImgDisDense_29")(lyr28)
    lyr30 = Dense(int(mid_disn/32), activation=hid_cls_act,
                  name="ImgDisDense_30")(lyr29)

    # output layer
    out_cls = Dense(out_nsize, activation=out_lyr_act, name="ImgDisOut")(lyr30)

    # MODEL DEFINITION
    model = Model(inputs=in_img, outputs=out_cls, name=dis_model_name)
    return model


def create_img_gan(gen_model, dis_model, gan_cfg):

    # getting GAN Config
    gan_model_name = gan_cfg.get("gan_model_name")
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
    model = Model(gen_noise, gan_output, name=gan_model_name)
    # compile model
    model.compile(loss=ls, optimizer=opt, metrics=met)
    # model.compile(loss=ls, optimizer=opt)
    return model

# ## Text GAN

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
                   name="TxtGenMask_1")(in_latent)  # concat1

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

    out_txt = TimeDistributed(
        Dense(txt_shape, activation=out_lyr_act), name="GenTxtOut")(lyr14)

    # model definition
    model = Model(inputs=in_latent, outputs=out_txt)

    return model


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
                   name="TxtDisMask_1")(in_txt)  # concat1

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
    lyr10 = Dense(int(mid_disn), activation=hid_cls_act,
                  name="TxtDisDense_10")(lyr9)
    lyr11 = Dense(int(mid_disn/2), activation=hid_cls_act,
                  name="TxtDisDense_11")(lyr10)
    # drop layer
    lyr12 = Dropout(hid_ldrop, name="TxtDisDrop_12")(lyr11)

    # dense classifier layers
    lyr13 = Dense(int(mid_disn/4), activation=hid_cls_act,
                  name="TxtDisDense_13")(lyr12)
    lyr14 = Dense(int(mid_disn/8), activation=hid_cls_act,
                  name="TxtDisDense_14")(lyr13)
    # drop layer
    lyr15 = Dropout(hid_ldrop, name="TxtDisDrop_15")(lyr14)

    # dense classifier layers
    lyr16 = Dense(int(mid_disn/16), activation=hid_cls_act,
                  name="TxtDisDense_16")(lyr15)
    lyr17 = Dense(int(mid_disn/32), activation=hid_cls_act,
                  name="TxtDisDense_17")(lyr16)

    # output layer
    out_cls = Dense(out_nsize, activation=out_lyr_act, name="TxtDisOut")(lyr17)

    # MODEL DEFINITION
    model = Model(inputs=in_txt, outputs=out_cls)
    return model


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

# ## Conditional Img GAN: CGAN-img

# convolutional generator for images


def create_img_cgenerator(latent_dims, n_labels, model_cfg):

    # MODEL CONFIG
    # config for conditional labels
    gen_model_name = model_cfg.get("gen_model_name")
    lbl_ly_actf = model_cfg.get("labels_lyr_activation")
    hid_ldrop = model_cfg.get("gen_dropout_rate")

    # def of the latent space size for the input
    latent_features = model_cfg.get("latent_features")
    latent_filters = model_cfg.get("latent_filters")
    latent_dense = latent_features*latent_features*latent_filters
    in_lyr_act = model_cfg.get("input_lyr_activation")
    latent_img_shape = model_cfg.get("latent_img_shape")
    latent_img_size = model_cfg.get("latent_img_size")

    # hidden layer config
    filters = model_cfg.get("filters")
    ksize = model_cfg.get("kernel_size")
    stsize = model_cfg.get("stride")
    pad = model_cfg.get("padding")
    hid_lyr_act = model_cfg.get("hidden_lyr_activation")

    # output layer condig
    out_filters = model_cfg.get("output_filters")
    out_ksize = model_cfg.get("output_kernel_size")
    out_stsize = model_cfg.get("output_stride")
    out_pad = model_cfg.get("output_padding")
    img_shape = model_cfg.get("output_shape")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # kernet initialization config
    initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    batchep = 0.00001

    # CONDITIONAL LABELS LAYERS
    # label input
    in_labels = Input(shape=(n_labels,), name="ImgCGenLblIn")

    # dense layer
    con3 = Dense(latent_img_size, activation=lbl_ly_actf,
                 name="ImgCGenLblDense_3")(in_labels)

    # batch normalization + drop layers to avoid overfit
    con4 = BatchNormalization(name="ImgCGenLblBN_4",
                              epsilon=batchep)(con3)
    con5 = Dropout(hid_ldrop, name="ImgCGenLblDrop_5")(con4)

    # reshape layer 1D-> 2D (rbg image)
    out_con = Reshape(latent_img_shape, name="ImgCGenLblOut")(con5)

    # LAYER CREATION
    # input layer
    in_latent = Input(shape=latent_dims, name="ImgCGenIn")

    # dense layer
    lyr1 = Dense(latent_dense,
                 activation=hid_lyr_act,
                 name="ImgCGenDense_1")(in_latent)

    # reshape layer 1D-> 2D (rbg image)
    lyr2 = Reshape(latent_img_shape, name="ImgCGenReshape_2")(lyr1)

    # concat generator layer + labels layer
    lbl_concat = Concatenate(axis=-1, name="ImgCGenConcat")([lyr2, out_con])

    # transpose conv2D layer
    lyr3 = Conv2DTranspose(int(filters), kernel_size=ksize,
                           kernel_initializer=initializer,
                           strides=stsize, activation=hid_lyr_act,
                           padding=pad, name="ImgCGenConv2D_3")(lbl_concat)

    # batch normalization + drop layers to avoid overfit
    lyr4 = BatchNormalization(name="ImgCGenBN_4",
                              epsilon=batchep)(lyr3)
    lyr5 = Dropout(hid_ldrop, name="ImgCGenDrop_5")(lyr4)

    # transpose conv2D layer
    lyr6 = Conv2DTranspose(int(filters/2), kernel_size=ksize,
                           kernel_initializer=initializer,
                           strides=stsize, activation=hid_lyr_act,
                           padding=pad, name="ImgCGenConv2D_6")(lyr5)

    # batch normalization + drop layers to avoid overfit
    lyr7 = BatchNormalization(name="ImgCGenBN_7",
                              epsilon=batchep)(lyr6)
    lyr8 = Dropout(hid_ldrop, name="ImgCGenDrop_8")(lyr7)

    # transpose conv2D layer
    lyr9 = Conv2DTranspose(int(filters/4), kernel_size=ksize,
                           kernel_initializer=initializer,
                           strides=stsize, activation=out_lyr_act,
                           padding=pad, name="ImgCGenConv2D_9")(lyr8)

    # batch normalization + drop layers to avoid overfit
    lyr10 = BatchNormalization(name="ImgCGenBN_10",
                               epsilon=batchep)(lyr9)
    lyr11 = Dropout(hid_ldrop, name="ImgCGenDrop_11")(lyr10)

    # transpose conv2D layer
    lyr12 = Conv2DTranspose(int(filters/8), kernel_size=ksize,
                            kernel_initializer=initializer,
                            strides=stsize, activation=hid_lyr_act,
                            padding=pad, name="ImgCGenConv2D_123")(lyr11)

    # batch normalization + drop layers to avoid overfit
    lyr13 = BatchNormalization(name="ImgCGenBN_13",
                               epsilon=batchep)(lyr12)
    lyr14 = Dropout(hid_ldrop, name="ImgCGenDrop_14")(lyr13)

    # transpose conv2D layer
    lyr15 = Conv2DTranspose(int(filters/16), kernel_size=out_ksize,
                            kernel_initializer=initializer,
                            strides=out_stsize, activation=hid_lyr_act,
                            padding=pad, name="ImgCGenConv2D_15")(lyr14)

    # batch normalization + drop layers to avoid overfit
    lyr16 = BatchNormalization(name="ImgCGenBN_16",
                               epsilon=batchep)(lyr15)
    lyr17 = Dropout(hid_ldrop, name="ImgCGenDrop_17")(lyr16)

    # # transpose conv2D layer
    # lyr18 = Conv2DTranspose(int(filters/32), kernel_size=ksize,
    #                         kernel_initializer=initializer,
    #                         strides=stsize, activation=hid_lyr_act,
    #                         padding=pad, name="ImgCGenConv2D_18")(lyr17)

    # # batch normalization + drop layers to avoid overfit
    # lyr19 = BatchNormalization(name="ImgCGenBN_19",
    #                             epsilon=batchep)(lyr18)
    # lyr20 = Dropout(hid_ldrop, name="ImgCGenDrop_20")(lyr19)

    # output layer
    out_img = Conv2D(out_filters, kernel_size=out_ksize,
                     kernel_initializer=initializer,
                     strides=out_stsize, activation=out_lyr_act,
                     padding=out_pad, input_shape=img_shape,
                     name="ImgCGenOut")(lyr17)

    # MODEL DEFINITION
    model = Model(inputs=[in_latent, in_labels],
                  outputs=out_img, name=gen_model_name)
    return model


# convolutional discriminator for images
def create_img_cdiscriminator(img_shape, n_labels, model_cfg):

    # MODEL CONFIG
    # config for conditional labels
    dis_model_name = model_cfg.get("dis_model_name")
    lbl_ly_actf = model_cfg.get("labels_lyr_activation")
    lbl_filters = model_cfg.get("labels_filters")
    lbl_ksize = model_cfg.get("labels_kernel_size")
    lbl_stsize = model_cfg.get("labels_stride")
    hid_ldrop = model_cfg.get("dis_dropout_rate")
    latent_img_size = model_cfg.get("latent_img_size")
    latent_img_shape = model_cfg.get("latent_img_shape")

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
    # mid neuron size
    mid_disn = model_cfg.get("mid_dis_neurons")
    hid_cls_act = model_cfg.get("dense_cls_activation")

    # output layer condig
    out_nsize = model_cfg.get("output_dis_neurons")
    out_lyr_act = model_cfg.get("output_lyr_activation")

    # kernet initialization config
    initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    batchep = 0.00001

    # CONDITIONAL LABELS LAYERS
    # label input
    in_labels = Input(shape=(n_labels,), name="ImgCDisLblIn")

    # dense layer
    con1 = Dense(latent_img_size, activation=lbl_ly_actf,
                 name="ImgCDisLblDense_2")(in_labels)

    # batch normalization + drop layers to avoid overfit
    con2 = BatchNormalization(name="ImgCDisLblBN_4",
                              epsilon=batchep)(con1)
    con3 = Dropout(hid_ldrop, name="ImgCDisLblDrop_5")(con2)

    # reshape layer 1D-> 2D (rbg image)
    con4 = Reshape(latent_img_shape, name="ImgCDisReshape_6")(con3)

    # transpose conv2D layer
    con5 = Conv2DTranspose(int(filters/2), kernel_size=lbl_ksize,
                           kernel_initializer=initializer,
                           strides=lbl_stsize, activation=lbl_ly_actf,
                           padding=pad, name="ImgCDisLblConv2D_5")(con4)

    # batch normalization + drop layers to avoid overfit
    con6 = BatchNormalization(name="ImgCDisLblBN_6",
                              epsilon=batchep)(con5)
    con7 = Dropout(hid_ldrop, name="ImgCDisLblDrop_7")(con6)

    # transpose conv2D layer
    con8 = Conv2DTranspose(int(filters/4), kernel_size=lbl_ksize,
                           kernel_initializer=initializer,
                           strides=lbl_stsize, activation=lbl_ly_actf,
                           padding=pad, name="ImgCDisLblDrop_8")(con7)

    # batch normalization + drop layers to avoid overfit
    con9 = BatchNormalization(name="ImgCDisLblBN_9",
                              epsilon=batchep)(con8)
    con10 = Dropout(hid_ldrop, name="ImgCDisLblDrop_10")(con9)

    # transpose conv2D layer
    con11 = Conv2DTranspose(int(filters/8), kernel_size=lbl_ksize,
                            kernel_initializer=initializer,
                            strides=lbl_stsize, activation=lbl_ly_actf,
                            padding=pad, name="ImgCDisLblConv2D_11")(con10)

    # batch normalization + drop layers to avoid overfit
    con12 = BatchNormalization(name="ImgCDisLblBN_12",
                               epsilon=batchep)(con11)
    con13 = Dropout(hid_ldrop, name="ImgCDisLblDrop_13")(con12)

    # transpose conv2D layer
    con14 = Conv2DTranspose(int(filters/16), kernel_size=lbl_ksize,
                            kernel_initializer=initializer,
                            strides=lbl_stsize, activation=lbl_ly_actf,
                            padding=pad, name="ImgCDisLblConv2D_14")(con13)

    # batch normalization + drop layers to avoid overfit
    con15 = BatchNormalization(name="ImgCDisLblBN_15",
                               epsilon=batchep)(con14)
    con16 = Dropout(hid_ldrop, name="ImgCDisLblDrop_16")(con15)

    # # transpose conv2D layer
    # con17 = Conv2DTranspose(int(filters/32), kernel_size=ksize,
    #                         kernel_initializer=initializer,
    #                         strides=stsize, activation=hid_lyr_act,
    #                         padding=pad, name="ImgCDisLblConv2D_17")(con16)

    # # batch normalization + drop layers to avoid overfit
    # con18 = BatchNormalization(name="ImgCDisLblBN_18",
    #                             epsilon=batchep)(con17)
    # con19 = Dropout(hid_ldrop, name="ImgCDisLblDrop_19")(con18)

    # output layer
    con_img = Conv2D(img_shape[2], kernel_size=(3, 3),
                     kernel_initializer=initializer,
                     strides=(1, 1), activation=lbl_ly_actf,
                     padding=pad, input_shape=img_shape,
                     name="ImgCDisLblOut")(con16)

    # LAYER CREATION
    # input layer
    in_img = Input(shape=img_shape, name="CDisImgIn")

    # concatenate in img + labels layer
    lbl_concat = Concatenate(axis=-1, name="ImgCDisConcat")([in_img, con_img])

    # DISCRIMINATOR LAYERS
    # intermediate conv layer 64 filters
    lyr1 = Conv2D(int(in_filters/64), kernel_size=in_ksize,
                  kernel_initializer=initializer,
                  padding=in_pad, activation=in_lyr_act,
                  strides=in_stsize, name="ImgCDisConv2D_1")(lbl_concat)

    # batch normalization + drop layers to avoid overfit
    lyr2 = BatchNormalization(name="ImgCDisBN_2",
                              epsilon=batchep)(lyr1)
    lyr3 = Dropout(hid_ldrop, name="ImgCDisDrop_3")(lyr2)

    # intermediate conv layer 128 filters
    lyr4 = Conv2D(int(in_filters/32), kernel_size=ksize,
                  kernel_initializer=initializer,
                  padding=pad, activation=hid_lyr_act,
                  strides=stsize, name="ImgCDisConv2D_4")(lyr3)

    # batch normalization + drop layers to avoid overfit
    lyr5 = BatchNormalization(name="ImgCDisBN_5",
                              epsilon=batchep)(lyr4)
    lyr6 = Dropout(hid_ldrop, name="ImgCDisDrop_6")(lyr5)

    # intermediate conv layer 256 filters
    sp_stsize = (1, 1)
    lyr7 = Conv2D(int(in_filters/16), kernel_size=ksize,
                  kernel_initializer=initializer,
                  padding=pad, activation=hid_lyr_act,
                  strides=sp_stsize, name="ImgCDisConv2D_7")(lyr6)

    # batch normalization + drop layers to avoid overfit
    lyr8 = BatchNormalization(name="ImgCDisBN_8",
                              epsilon=batchep)(lyr7)
    lyr9 = Dropout(hid_ldrop, name="ImgCDisDrop_9")(lyr8)

    # intermediate conv layer 512 filters
    lyr10 = Conv2D(int(filters/8), kernel_size=ksize,
                   kernel_initializer=initializer,
                   padding=pad, activation=hid_lyr_act,
                   strides=stsize, name="ImgCDisConv2D_10")(lyr9)

    # batch normalization + drop layers to avoid overfit
    lyr11 = BatchNormalization(name="ImgCDisBN_11",
                               epsilon=batchep)(lyr10)
    lyr12 = Dropout(hid_ldrop, name="ImgCDisDrop_12")(lyr11)

    # intermediate conv layer 1024 filters
    lyr13 = Conv2D(int(filters/4), kernel_size=ksize,
                   kernel_initializer=initializer,
                   padding=pad, activation=hid_lyr_act,
                   strides=stsize, name="ImgCDisConv2D_13")(lyr12)

    # batch normalization + drop layers to avoid overfit
    lyr14 = BatchNormalization(name="ImgCDisBN_14",
                               epsilon=batchep)(lyr13)
    lyr15 = Dropout(hid_ldrop, name="ImgCDisDrop_15")(lyr14)

    # intermediate conv layer
    lyr16 = Conv2D(int(filters/2), kernel_size=ksize,
                   padding=pad, activation=hid_lyr_act,
                   strides=stsize, name="ImgCDisConv2D_16")(lyr15)

    # batch normalization + drop layers to avoid overfit
    lyr17 = BatchNormalization(name="ImgCDisBN_17",
                               epsilon=batchep)(lyr16)
    lyr18 = Dropout(hid_ldrop, name="ImgCDisDrop_18")(lyr17)

    # # intermediate conv layer
    # lyr19 = Conv2D(int(filters), kernel_size=ksize,
    #                 padding=pad, activation=hid_lyr_act,
    #                 strides=stsize, name="ImgCDisConv2D_19")(lyr18)

    # # batch normalization + drop layers to avoid overfit
    # lyr20 = BatchNormalization(name="ImgCDisBN_20",
    #                             epsilon=batchep)(lyr19)
    # lyr21 = Dropout(hid_ldrop, name="ImgCDisDrop_21")(lyr20)

    # flatten from 2D to 1D
    lyr22 = Flatten(name="ImgCDisFlat_22")(lyr18)

    # dense classifier layers
    lyr23 = Dense(int(mid_disn), activation=hid_cls_act,
                  name="ImgCDisDense_23")(lyr22)
    lyr24 = Dense(int(mid_disn/2), activation=hid_cls_act,
                  name="ImgCDisDense_24")(lyr23)
    # drop layer
    lyr25 = Dropout(hid_ldrop, name="ImgCDisDrop_25")(lyr24)

    # dense classifier layers
    lyr26 = Dense(int(mid_disn/4), activation=hid_cls_act,
                  name="ImgCDisDense_26")(lyr25)
    lyr27 = Dense(int(mid_disn/8), activation=hid_cls_act,
                  name="ImgCDisDense_27")(lyr26)
    # drop layer
    lyr28 = Dropout(hid_ldrop, name="ImgCDisDrop_28")(lyr27)

    # dense classifier layers
    lyr29 = Dense(int(mid_disn/16), activation=hid_cls_act,
                  name="ImgCDisDense_29")(lyr28)
    lyr30 = Dense(int(mid_disn/32), activation=hid_cls_act,
                  name="ImgCDisDense_30")(lyr29)

    # output layer
    out_cls = Dense(out_nsize, activation=out_lyr_act,
                    name="ImgCDisOut")(lyr30)

    # MODEL DEFINITION
    model = Model(inputs=[in_img, in_labels],
                  outputs=out_cls, name=dis_model_name)
    return model


def create_img_cgan(gen_model, dis_model, gan_cfg):

    # getting GAN Config
    gan_model_name = gan_cfg.get("gan_model_name")
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
    gan_model = Model([gen_noise, gen_labels], gan_output, name=gan_model_name)
    # compile model
    gan_model.compile(loss=ls, optimizer=opt, metrics=met)
    # cgan_model.compile(loss=gan_cfg[0], optimizer=gan_cfg[1])#, metrics=gan_cfg[2])
    return gan_model

# ## Multi GAN txt2img

# LSTM + Conv discriminator for image and text
# TODO need to implement this


def create_multi_discriminator(img_shape, txt_shape, model_cfg):

    # model definition
    in_img = None
    in_txt = None
    out_cls = None
    model = Model(inputs=[in_img, in_txt], outputs=out_cls)
    return model


# TODO need to implement this
def create_multi_generator(latent_dims, img_shape, txt_shape, model_cfg):

    # model definition
    in_latent = None
    out_img = None
    out_txt = None
    model = Model(inputs=in_latent, outputs=[out_img, out_txt])
    return model


# TODO need to implement this
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
    gan_model = Model([gen_noise, gen_labels], gan_output)
    # compile model
    gan_model.compile(
        loss=gan_cfg[0], optimizer=gan_cfg[1], metrics=gan_cfg[2])
    # cgan_model.compile(loss=gan_cfg[0], optimizer=gan_cfg[1])#, metrics=gan_cfg[2])
    return gan_model

# ## Multi CGAN txt2img


def create_multi_cgenerator(latent_dims, img_shape, txt_shape, n_labels, model_cfg):

    # MODEL CONFIG
    # config for conditional labels
    # print("=======================\n",model_cfg, "=====================")
    gen_model_name = model_cfg.get("gen_model_name")
    memory = model_cfg.get("memory")
    features = model_cfg.get("features")
    lbl_ly_actf = model_cfg.get("labels_lyr_activation")
    hid_ldrop = model_cfg.get("gen_dropout_rate")

    # def of the latent space size for the input
    # input layer config, latent txt space
    latent_features = model_cfg.get("latent_features")
    latent_filters = model_cfg.get("latent_filters")
    latent_img_dense = latent_features*latent_features*latent_filters
    latent_txt_dense = memory*features
    latent_img_size = model_cfg.get("latent_img_size")
    latent_img_shape = model_cfg.get("latent_img_shape")
    latent_txt_size = model_cfg.get("latent_txt_size")
    latent_txt_shape = model_cfg.get("latent_txt_shape")
    # latent_ntxt = model_cfg.get("mid_gen_neurons")

    in_lyr_act = model_cfg.get("input_lyr_activation")
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
    out_rs = model_cfg.get("output_return_sequences")

    # kernet initialization config
    initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    batchep = 0.00001

    ############################## OJO LABELS START ##################################

    # CONDITIONAL LABELS LAYERS
    # label input
    in_labels = Input(shape=(n_labels,), name="MultiCGenLblIn")

    # image conditional layers
    # dense layer
    icon1 = Dense(latent_img_size, activation=lbl_ly_actf,
                  name="MultiImgCGenLblDense_3")(in_labels)

    # batch normalization + drop layers to avoid overfit
    icon2 = BatchNormalization(name="MultiImgCGenLblBN_4",
                               epsilon=batchep)(icon1)
    icon3 = Dropout(hid_ldrop, name="MultiImgCGenLblDrop_5")(icon2)

    # reshape layer 1D-> 2D (rbg image)
    iout_con = Reshape(latent_img_shape, name="MultiImgCGenLblOut")(icon3)

    # text conditional layers
    # dense layer
    tcon1 = Dense(latent_txt_size, activation=lbl_ly_actf,
                  name="MultiTxtCGenLblDense_3")(in_labels)

    # batch normalization + drop layers to avoid overfit
    tcon2 = BatchNormalization(name="MultiTxtCGenLblBN_4",
                               epsilon=batchep)(tcon1)
    tcon3 = Dropout(hid_ldrop, name="MultiTxtCGenLblDrop_5")(tcon2)

    # reshape layer 1D-> 2D (rbg image)
    tout_con = Reshape(latent_txt_shape, name="MultiTxtCGenLblOut")(tcon3)

    # LAYER CREATION
    # input layer
    in_latent = Input(shape=latent_dims, name="ImgMultiCGenIn")

    # dense layer for rgb image
    lyr1 = Dense(latent_img_dense,
                 activation=in_lyr_act,
                 name="ImgMultiGenDense_1")(in_latent)

    # dense layer for text data
    lyr2 = Dense(latent_txt_dense,
                 activation=in_lyr_act,
                 name="TxtMultiGenDense_2")(in_latent)

    # RGB IMAGE GENERATOR
    #  reshape layer 1D-> 2D (rbg image)
    ilyr2 = Reshape(latent_img_shape, name="ImgMultiCGenReshape_2")(lyr1)

    # concat generator layer + labels layer
    ilbl_concat = Concatenate(
        axis=-1, name="ImgMultiCGenConcat")([ilyr2, iout_con])

    # transpose conv2D layer
    ilyr3 = Conv2DTranspose(int(filters), kernel_size=ksize,
                            kernel_initializer=initializer,
                            strides=stsize, activation=hid_lyr_act,
                            padding=pad, name="ImgMultiCGenConv2D_3")(ilbl_concat)

    # batch normalization + drop layers to avoid overfit
    ilyr4 = BatchNormalization(name="ImgMultiGenBN_4",
                               epsilon=batchep)(ilyr3)
    ilyr5 = Dropout(hid_ldrop, name="ImgMultiCGenDrop_5")(ilyr4)

    # transpose conv2D layer
    ilyr6 = Conv2DTranspose(int(filters/2), kernel_size=ksize,
                            kernel_initializer=initializer,
                            strides=stsize, activation=hid_lyr_act,
                            padding=pad, name="ImgMultiCGenConv2D_6")(ilyr5)

    # batch normalization + drop layers to avoid overfit
    ilyr7 = BatchNormalization(name="ImgMultiCGenBN_7",
                               epsilon=batchep)(ilyr6)
    ilyr8 = Dropout(hid_ldrop, name="ImgMultiCGenDrop_8")(ilyr7)

    # transpose conv2D layer
    ilyr9 = Conv2DTranspose(int(filters/4), kernel_size=ksize,
                            kernel_initializer=initializer,
                            strides=stsize, activation=hid_lyr_act,
                            padding=pad, name="ImgMultiCGenConv2D_9")(ilyr8)

    # batch normalization + drop layers to avoid overfit
    ilyr10 = BatchNormalization(name="ImgMultiCGenBN_10",
                                epsilon=batchep)(ilyr9)
    ilyr11 = Dropout(hid_ldrop, name="ImgMultiCGenDrop_11")(ilyr10)

    # transpose conv2D layer
    ilyr12 = Conv2DTranspose(int(filters/8), kernel_size=ksize,
                             kernel_initializer=initializer,
                             strides=stsize, activation=hid_lyr_act,
                             padding=pad, name="ImgMultiCGenConv2D_13")(ilyr11)

    # batch normalization + drop layers to avoid overfit
    ilyr13 = BatchNormalization(name="ImgMultiCGenBN_13",
                                epsilon=batchep)(ilyr12)
    ilyr14 = Dropout(hid_ldrop, name="ImgMultiCGenDrop_14")(ilyr13)

    # transpose conv2D layer
    ilyr15 = Conv2DTranspose(int(filters/16), kernel_size=out_ksize,
                             kernel_initializer=initializer,
                             strides=out_stsize, activation=hid_lyr_act,
                             padding=pad, name="ImgMultiCGenConv2D_15")(ilyr14)

    # batch normalization + drop layers to avoid overfit
    ilyr16 = BatchNormalization(name="ImgMultiCGenBN_16",
                                epsilon=batchep)(ilyr15)
    ilyr17 = Dropout(hid_ldrop, name="ImgMultiCGenDrop_17")(ilyr16)

    # transpose conv2D layer
    ilyr18 = Conv2DTranspose(int(filters/32), kernel_size=out_ksize,
                             kernel_initializer=initializer,
                             strides=out_stsize, activation=hid_lyr_act,
                             padding=pad, name="ImgMultiCGenConv2D_18")(ilyr17)

    # batch normalization + drop layers to avoid overfit
    ilyr19 = BatchNormalization(name="ImgMultiCGenBN_19",
                                epsilon=batchep)(ilyr18)
    ilyr20 = Dropout(hid_ldrop, name="ImgMultiCGenDrop_20")(ilyr19)

    # output layer
    out_img = Conv2D(out_filters, kernel_size=out_ksize,
                     kernel_initializer=initializer,
                     strides=out_stsize, activation=out_lyr_act,
                     padding=out_pad, input_shape=img_shape,
                     name="ImgMultiCGenOut")(ilyr20)

    # TEXT DATA GENERATOR
    # reshape layer 1D-> 2D (descriptive txt)
    tlyr3 = Reshape(latent_txt_shape, name="TxtMultiCGenReshape_3")(lyr2)

    # concat generator layer + labels layer
    tlbl_concat = Concatenate(
        axis=-1, name="TxtMultiCGenConcat")([tlyr3, tout_con])

    # masking input text
    tlyr4 = Masking(mask_value=mval, input_shape=mem_shape,
                    name="TxtMultiCGenMask_4")(tlbl_concat)

    # intermediate recurrent layer
    tlyr5 = LSTM(int(lstm_units/4), activation=hid_lyr_act,
                 kernel_initializer=initializer,
                 input_shape=mem_shape,
                 return_sequences=rs,
                 name="TxtMultiCGenLSTM_5")(tlyr4)

    # batch normalization + drop layers to avoid overfit
    tlyr6 = BatchNormalization(name="TxtMultiCGenBN_6",
                               epsilon=batchep)(tlyr5)
    tlyr7 = Dropout(hid_ldrop, name="TxtMultiCGenDrop_7")(tlyr6)

    # intermediate recurrent layer
    tlyr8 = LSTM(int(lstm_units/2), activation=hid_lyr_act,
                 kernel_initializer=initializer,
                 input_shape=mem_shape,
                 return_sequences=rs,
                 name="TxtMultiCGenLSTM_8")(tlyr7)

    # batch normalization + drop layers to avoid overfit
    tlyr9 = BatchNormalization(name="TxtMultiCGenBN_9",
                               epsilon=batchep)(tlyr8)
    tlyr10 = Dropout(hid_ldrop, name="TxtMultiCGenDrop_10")(tlyr9)

    # output layer, dense time sequential layer.
    tlyr11 = LSTM(lstm_units, activation=hid_lyr_act,
                  kernel_initializer=initializer,
                  input_shape=mem_shape,
                  return_sequences=rs,
                  name="TxtMultiCGenLSTM_11")(tlyr10)

    out_txt = TimeDistributed(
        Dense(txt_shape, activation=out_lyr_act), name="TxtMultiCGenOut")(tlyr11)

    # MODEL DEFINITION
    model = Model(inputs=[in_latent, in_labels], outputs=[
                  out_img, out_txt], name=gen_model_name)

    return model


# LSTM + Conv conditianal discriminator for text and images
def create_multi_cdiscriminator(img_shape, txt_shape, n_labels, model_cfg):

    # print("=======================\n", model_cfg, "\n")
    # MODEL CONFIG
    # config for txt + img conditional label
    dis_model_name = model_cfg.get("dis_model_name")
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

    # latent_img_dense = latent_features*latent_features*latent_filters
    # latent_txt_dense = memory*features
    latent_img_size = model_cfg.get("latent_img_size")
    # latent_img_shape = model_cfg.get("latent_img_shape")
    latent_txt_size = model_cfg.get("latent_txt_size")
    # latent_txt_shape = model_cfg.get("latent_txt_shape")

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

    # kernet initialization config
    initializer = tf.keras.initializers.RandomNormal(stddev=0.02)
    batchep = 0.00001

    # CONDITIONAL LABELS LAYERS
    # label input
    in_labels = Input(shape=(n_labels,), name="ImgCDisLblIn")

    # image conditional layers
    # dense layer
    icon1 = Dense(latent_img_size, activation=lbl_ly_actf,
                  name="ImgCDisLblDense_1")(in_labels)

    # batch normalization + drop layers to avoid overfit
    icon2 = BatchNormalization(name="ImgCDisLblBN_2",
                               epsilon=batchep)(icon1)
    icon3 = Dropout(hid_ldrop, name="ImgCDisLblDrop_3")(icon2)

    # reshape layer 1D-> 2D (rbg image)
    icon4 = Reshape(dis_img_reshape, name="ImgCDisReshape_4")(icon3)

    # transpose conv2D layer
    icon5 = Conv2DTranspose(int(filters/2), kernel_size=lbl_ksize,
                            kernel_initializer=initializer,
                            strides=lbl_stsize, activation=lbl_ly_actf,
                            padding=pad, name="ImgCDisLblConv2D_5")(icon4)

    # batch normalization + drop layers to avoid overfit
    icon6 = BatchNormalization(name="ImgCDisLblBN_6",
                               epsilon=batchep)(icon5)
    icon7 = Dropout(hid_ldrop, name="ImgCDisLblDrop_7")(icon6)

    # transpose conv2D layer
    icon8 = Conv2DTranspose(int(filters/4), kernel_size=lbl_ksize,
                            kernel_initializer=initializer,
                            strides=lbl_stsize, activation=lbl_ly_actf,
                            padding=pad, name="ImgCDisLblDrop_8")(icon7)

    # batch normalization + drop layers to avoid overfit
    icon9 = BatchNormalization(name="ImgCDisLblBN_9",
                               epsilon=batchep)(icon8)
    icon10 = Dropout(hid_ldrop, name="ImgCDisLblDrop_10")(icon9)

    # transpose conv2D layer
    icon11 = Conv2DTranspose(int(filters/8), kernel_size=lbl_ksize,
                             kernel_initializer=initializer,
                             strides=lbl_stsize, activation=lbl_ly_actf,
                             padding=pad, name="ImgCDisLblConv2D_11")(icon10)

    # batch normalization + drop layers to avoid overfit
    icon12 = BatchNormalization(name="ImgCDisLblBN_12",
                                epsilon=batchep)(icon11)
    icon13 = Dropout(hid_ldrop, name="ImgCDisLblDrop_13")(icon12)

    # transpose conv2D layer
    icon14 = Conv2DTranspose(int(filters/16), kernel_size=lbl_ksize,
                             kernel_initializer=initializer,
                             strides=lbl_stsize, activation=lbl_ly_actf,
                             padding=pad, name="ImgCDisLblConv2D_14")(icon13)

    # batch normalization + drop layers to avoid overfit
    icon15 = BatchNormalization(name="ImgCDisLblBN_15",
                                epsilon=batchep)(icon14)
    icon16 = Dropout(hid_ldrop, name="ImgCDisLblDrop_16")(icon15)

    # output layer
    icon_out = Conv2D(img_shape[2], kernel_size=(3, 3),
                      kernel_initializer=initializer,
                      strides=(1, 1), activation=lbl_ly_actf,
                      padding=pad, input_shape=img_shape,
                      name="ImgMultiCDisLblOut")(icon16)

    # text conditional layers
    # dense layer
    tcon1 = Dense(latent_txt_size, activation=lbl_ly_actf,
                  name="MultiTxtCDisLblDense_1")(in_labels)

    # batch normalization + drop layers to avoid overfit
    tcon2 = BatchNormalization(name="MultiTxtCDisLblBN_2",
                               epsilon=batchep)(tcon1)
    tcon3 = Dropout(hid_ldrop, name="MultiTxtCDisLblDrop_3")(tcon2)

    # reshape layer 1D-> 2D (descriptive txt)
    tcon4 = Reshape(dis_txt_reshape, name="MultiTxtCDisReshape_4")(tcon3)

    # TEXT DATA GENERATOR
    # masking input text
    tcon5 = Masking(mask_value=mval, input_shape=mem_shape,
                    name="TxtMultiCDisMask_5")(tcon4)

    # intermediate recurrent layer
    tcon6 = LSTM(int(lbl_lstm/4), activation=lbl_ly_actf,
                 kernel_initializer=initializer,
                 input_shape=mem_shape,
                 return_sequences=lbl_rs,
                 name="TxtMultiCDisLSTM_6")(tcon5)

    # batch normalization + drop layers to avoid overfit
    tcon7 = BatchNormalization(name="TxtMultiCDisBN_7",
                               epsilon=batchep)(tcon6)
    tcon8 = Dropout(hid_ldrop, name="TxtMultiCDisDrop_8")(tcon7)

    # intermediate recurrent layer
    tcon9 = LSTM(int(lbl_lstm/2), activation=lbl_ly_actf,
                 kernel_initializer=initializer,
                 input_shape=mem_shape,
                 return_sequences=lbl_rs,
                 name="TxtMultiCDisLSTM_9")(tcon8)

    # batch normalization + drop layers to avoid overfit
    tcon10 = BatchNormalization(name="TxtMultiCGenBN_10",
                                epsilon=batchep)(tcon9)
    tcon11 = Dropout(hid_ldrop, name="TxtMultiCDisDrop_11")(tcon10)

    # intermediate recurrent layer
    tcon12 = LSTM(lbl_lstm, activation=lbl_ly_actf,
                  kernel_initializer=initializer,
                  input_shape=mem_shape,
                  return_sequences=lbl_rs,
                  name="TxtMultiCDisLSTM_12")(tcon11)

    # output layer, dense time sequential layer.
    # print(txt_shape, type(txt_shape))
    tcon_out = TimeDistributed(
        Dense(txt_shape[1], activation=lbl_ly_actf), name="TxtMultiCDisLblOut")(tcon12)

    # LAYER CREATION
    # IMAGE DISCRIMINATOR
    # input layer
    in_img = Input(shape=img_shape, name="ImgMulitCDisIn")

    # concatenate in img + labels layer
    lbl_concat = Concatenate(
        axis=-1, name="ImgMultiCDisConcat")([in_img, icon_out])

    # DISCRIMINATOR LAYERS
    # intermediate conv layer 64 filters
    ilyr1 = Conv2D(int(in_filters/64), kernel_size=in_ksize,
                   kernel_initializer=initializer,
                   padding=in_pad, activation=in_lyr_act,
                   strides=in_stsize, name="ImgMultiCDisConv2D_1")(lbl_concat)

    # batch normalization + drop layers to avoid overfit
    ilyr2 = BatchNormalization(name="ImgMultiCDisBN_2",
                               epsilon=batchep)(ilyr1)
    ilyr3 = Dropout(hid_ldrop, name="ImgMultiCDisDrop_3")(ilyr2)

    # intermediate conv layer 128 filters
    ilyr4 = Conv2D(int(in_filters/32), kernel_size=ksize,
                   kernel_initializer=initializer,
                   padding=pad, activation=hid_lyr_act,
                   strides=stsize, name="ImgMultiCDisConv2D_4")(ilyr3)

    # batch normalization + drop layers to avoid overfit
    ilyr5 = BatchNormalization(name="ImgMultiCDisBN_5",
                               epsilon=batchep)(ilyr4)
    ilyr6 = Dropout(hid_ldrop, name="ImgMultiCDisDrop_6")(ilyr5)

    # intermediate conv layer 256 filters
    sp_stsize = (1, 1)
    ilyr7 = Conv2D(int(in_filters/16), kernel_size=ksize,
                   kernel_initializer=initializer,
                   padding=pad, activation=hid_lyr_act,
                   strides=sp_stsize, name="ImgMultiCDisConv2D_7")(ilyr6)

    # batch normalization + drop layers to avoid overfit
    ilyr8 = BatchNormalization(name="ImgMultiCDisBN_8",
                               epsilon=batchep)(ilyr7)
    ilyr9 = Dropout(hid_ldrop, name="ImgMultiCDisDrop_9")(ilyr8)

    # intermediate conv layer 512 filters
    ilyr10 = Conv2D(int(filters/8), kernel_size=ksize,
                    kernel_initializer=initializer,
                    padding=pad, activation=hid_lyr_act,
                    strides=stsize, name="ImgMultiCDisConv2D_10")(ilyr9)

    # batch normalization + drop layers to avoid overfit
    ilyr11 = BatchNormalization(name="ImgMultiCDisBN_11",
                                epsilon=batchep)(ilyr10)
    ilyr12 = Dropout(hid_ldrop, name="ImgMultiCDisDrop_12")(ilyr11)

    # intermediate conv layer 1024 filters
    ilyr13 = Conv2D(int(filters/4), kernel_size=ksize,
                    kernel_initializer=initializer,
                    padding=pad, activation=hid_lyr_act,
                    strides=stsize, name="ImgMultiCDisConv2D_13")(ilyr12)

    # batch normalization + drop layers to avoid overfit
    ilyr14 = BatchNormalization(name="ImgMultiCDisBN_14",
                                epsilon=batchep)(ilyr13)
    ilyr15 = Dropout(hid_ldrop, name="ImgMultiCDisDrop_15")(ilyr14)

    # intermediate conv layer
    ilyr16 = Conv2D(int(filters/2), kernel_size=ksize,
                    kernel_initializer=initializer,
                    padding=pad, activation=hid_lyr_act,
                    strides=stsize, name="ImgMultiCDisConv2D_16")(ilyr15)

    # batch normalization + drop layers to avoid overfit
    ilyr17 = BatchNormalization(name="ImgMultiCDisBN_17",
                                epsilon=batchep)(ilyr16)
    ilyr18 = Dropout(hid_ldrop, name="ImgMultiCDisDrop_18")(ilyr17)

    # flatten from 2D to 1D
    ilyr19 = Flatten(name="ImgMultiCDisFlat_19")(ilyr18)

    # dense text classifier layers
    ilyr20 = Dense(int(mid_disn), activation=hid_cls_act,
                   name="ImgMultiCDisDense_20")(ilyr19)
    ilyr21 = Dense(int(mid_disn/2), activation=hid_cls_act,
                   name="ImgMultiCDisDense21")(ilyr20)
    # drop text layer
    ilyr22 = Dropout(hid_ldrop, name="ImgMultiCDisDrop_22")(ilyr21)

    # dense text classifier layers
    ilyr23 = Dense(int(mid_disn/4), activation=hid_cls_act,
                   name="ImgMultiCDisDense_23")(ilyr22)
    ilyr24 = Dense(int(mid_disn/8), activation=hid_cls_act,
                   name="ImgMultiCDisDense_24")(ilyr23)
    # drop text layer
    ilyr25 = Dropout(hid_ldrop, name="ImgMultiCDisDrop_25")(ilyr24)

    # dense text classifier layers
    ilyr26 = Dense(int(mid_disn/16), activation=hid_cls_act,
                   name="ImgMultiCDisDense_26")(ilyr25)
    ilyr27 = Dense(int(mid_disn/32), activation=hid_cls_act,
                   name="ImgMultiCDisDense_27")(ilyr26)

    # output text layer
    iout_cls = Dense(out_nsize, activation=out_lyr_act,
                     name="ImgMultiCDisOut")(ilyr27)

    #TXT DISCRIMINATOR
    # LAYER CREATION
    # input layer
    in_txt = Input(shape=txt_shape, name="TxtMultiCDisIn")

    # concat txt input with labels conditional
    concat_txt = Concatenate(
        axis=-1, name="TxtMultiCDisConcat")([in_txt, tcon_out])

    # DISCRIMINATOR LAYERS
    # masking input text
    tlyr1 = Masking(mask_value=mval, input_shape=txt_shape,
                    name="TxtMultiCDisMask_1")(concat_txt)  # concat1

    # input LSTM layer
    tlyr2 = LSTM(in_lstm, activation=in_lyr_act,
                 kernel_initializer=initializer,
                 input_shape=txt_shape,
                 return_sequences=in_rs,
                 name="TxtMultiCDisLSTM_2")(tlyr1)

    # batch normalization + drop layers to avoid overfit
    tlyr3 = BatchNormalization(name="TxtMultiCDisBN_3",
                               epsilon=batchep)(tlyr2)
    tlyr4 = Dropout(hid_ldrop, name="TxtMultiCtDisDrop_4")(tlyr3)

    # intermediate LSTM layer
    tlyr5 = LSTM(int(lstm_units/2),
                 activation=hid_lyr_act,
                 kernel_initializer=initializer,
                 input_shape=mem_shape,
                 return_sequences=rs,
                 name="TxtMultiCDisLSTM_5")(tlyr4)

    # batch normalization + drop layers to avoid overfit
    tlyr6 = BatchNormalization(name="TxtMultiCDisBN_6",
                               epsilon=batchep)(tlyr5)
    tlyr7 = Dropout(hid_ldrop, name="TxtMultiCDisDrop_7")(tlyr6)

    # intermediate LSTM layer
    tlyr8 = LSTM(int(lstm_units/4),
                 kernel_initializer=initializer,
                 activation=hid_lyr_act,
                 input_shape=mem_shape,
                 return_sequences=rs,
                 name="TxtMultiCDisLSTM_8")(tlyr7)

    # batch normalization + drop layers to avoid overfit
    tlyr9 = BatchNormalization(name="TxtMultiCDisBN_9",
                               epsilon=batchep)(tlyr8)
    tlyr10 = Dropout(hid_ldrop, name="TxtMultiCDisDrop_10")(tlyr9)

    # flatten from 2D to 1D
    tlyr11 = Flatten(name="TxtMultiCDisFlat_11")(tlyr10)

    # dense text classifier layers
    tlyr12 = Dense(int(mid_disn/2), activation=hid_cls_act,
                   name="TxtMultiCDisDense_12")(tlyr11)
    tlyr13 = Dense(int(mid_disn/4), activation=hid_cls_act,
                   name="TxtMultiCDisDense_13")(tlyr12)
    # drop text layer
    tlyr14 = Dropout(hid_ldrop, name="TxtMultiCDisDrop_14")(tlyr13)

    # dense text classifier layers
    tlyr15 = Dense(int(mid_disn/8), activation=hid_cls_act,
                   name="TxtMultiCDisDense_15")(tlyr14)
    tlyr16 = Dense(int(mid_disn/16), activation=hid_cls_act,
                   name="TxtMultiCDisDense_16")(tlyr15)
    # drop text layer
    tlyr17 = Dropout(hid_ldrop, name="TxtMultiCDisDrop_17")(tlyr16)

    # dense text classifier layers
    tlyr18 = Dense(int(mid_disn/32), activation=hid_cls_act,
                   name="TxtMultiCDisDense_18")(tlyr17)
    tlyr19 = Dense(int(mid_disn/64), activation=hid_cls_act,
                   name="TxtMultiCDisDense_19")(tlyr18)

    # output text layer
    tout_cls = Dense(out_nsize, activation=out_lyr_act,
                     name="TxtMultiCDisOut")(tlyr19)

    # # concat img encoding + txt encoding
    # concat_encoding = Concatenate(axis=-1, name="MultiCDisDenseConcat")([ilyr19, tlyr11])

    # # dense classifier layers
    # lyr1 = Dense(int(mid_disn), activation=hid_cls_act, name="MultiCDisDense_1")(concat_encoding)
    # lyr2 = Dense(int(mid_disn/2), activation=hid_cls_act, name="MultiCDisDense_2")(lyr1)
    # # drop layer
    # lyr3 = Dropout(hid_ldrop, name="MultiCDisDrop_3")(lyr2)

    # # dense classifier layers
    # lyr4 = Dense(int(mid_disn/4), activation=hid_cls_act, name="MultiCDisDense_4")(lyr3)
    # lyr5 = Dense(int(mid_disn/8), activation=hid_cls_act, name="MultiCDisDense_5")(lyr4)
    # # drop layer
    # lyr6 = Dropout(hid_ldrop, name="MultiCDisDrop_6")(lyr5)

    # # dense classifier layers
    # lyr7 = Dense(int(mid_disn/16), activation=hid_cls_act, name="MultiCDisDense_7")(lyr6)
    # lyr8 = Dense(int(mid_disn/32), activation=hid_cls_act, name="MultiCDisDense_8")(lyr7)

    # # output layer
    # out_cls = Dense(out_nsize, activation=out_lyr_act, name="MultiCDisOut")(lyr8)

    # model definition
    model = Model(inputs=[in_img, in_txt, in_labels], outputs=[
                  iout_cls, tout_cls], name=dis_model_name)
    return model


def create_multi_cgan(gen_model, dis_model, gan_cfg):

    # getting GAN Config
    gan_model_name = gan_cfg.get("gan_model_name")
    ls = gan_cfg.get("loss")
    opt = gan_cfg.get("optimizer")
    met = gan_cfg.get("metrics")
    lw = gan_cfg.get("loss_weights")

    # make weights in the discriminator not trainable
    dis_model.trainable = False
    # get noise and label inputs from generator model
    gen_noise, gen_labels = gen_model.input
    # get image output from the generator model
    gen_img, gen_txt = gen_model.output
    # connect image output and label input from generator as inputs to discriminator
    img_cls, txt_cls = dis_model([gen_img, gen_txt, gen_labels])
    # define gan model as taking noise and label and outputting a classification
    gan_model = Model([gen_noise, gen_labels], [
                      img_cls, txt_cls], name=gan_model_name)
    # compile model
    gan_model.compile(loss=ls, optimizer=opt, metrics=met, loss_weights=lw)
    # cgan_model.compile(loss=gan_cfg[0], optimizer=gan_cfg[1])#, metrics=gan_cfg[2])
    return gan_model


# function to fromat lexicon/dictionary to translate for humnas
def format_tfidf_tokens(tfidf_tokens):

    tfidf_dict = list()

    for tfidf in tfidf_tokens:

        tfidf = eval(tfidf)
        # print(type(tfidf), tfidf)
        td = dict(tfidf)
        tfidf_dict.append(td)

    return tfidf_dict


# function to find a name of column names according to a regex
def get_keeper_cols(col_names, search_regex):
    ans = [i for i in col_names if re.search(search_regex, i)]
    return ans


# function to find the disperse columns in the df
def get_disperse_categories(src_df, keep_cols, max_dis, check_cols, ignore_col):

    ans = list()

    max_dis = 2
    tcount = 0

    while tcount < max_dis:
        for label_col in keep_cols:

            if label_col != ignore_col:

                label_count = src_df[label_col].value_counts(normalize=False)

                if tcount < label_count.shape[0] and (check_cols in label_col):
                    tcount = label_count.shape[0]
                    ans.append(label_col)
                # print("count values of", label_col, ":=", label_count.shape)#.__dict__)
        tcount = tcount + 1

    return ans


# function to remove the disperse columns from the interesting ones
def remove_disperse_categories(keep_columns, too_disperse):
    for too in too_disperse:
        keep_columns.remove(too)
    return keep_columns


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


def heat_categories(train_df, cat_cols, tgt_col):

    labels_data = train_df[cat_cols]
    labels_concat = list()

    # concatenating all category labels from dataframe
    for index, row in labels_data.iterrows():
        row = concat_labels(row, cat_cols)
        labels_concat.append(row)

    # print(len(labels_concat[0]), type(labels_concat[0]))
    # updating dataframe
    tcat_label_col = "std_cat_labels"
    train_df[tgt_col] = labels_concat

    return train_df


# function to adjust the textual data for the LSTM layers in the model
def format_corpus(corpus, timesteps, features):

    # preparation for reshape lstm model
    corpus = temporalize(corpus, timesteps)
    print(corpus.shape)

    corpus = corpus.reshape((corpus.shape[0], timesteps, features))
    print(corpus.shape)

    return corpus


# main of the program
if __name__ == "__main__":
    print("========= STARTING MAIN SCRIPT!!! ========")
    # creating the View() object and running it

    # ===================== CUDA CHECK =================

    # disable GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    # ckeking GPU availability
    # device_lib.list_local_devices()
    devices = tf.config.experimental.list_physical_devices()
    gpus = tf.config.experimental.list_physical_devices("GPU")

    print("Available Devices:\n", devices)
    print("Available GPUs:\n", gpus)

    # config GPUS
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

    # # checking size and use
    # for d in gpus:
    #     t = d.device_type
    #     # print(t)
    #     print(d)
    #     name = tf.config.get_logical_device_configuration(d)
    #     print(name)
    #     l = [item.split(':',1) for item in name.split(", ")]
    #     name_attr = dict([x for x in l if len(x)==2])
    #     dev = name_attr.get('name', 'Unnamed device')
    #     print(f" {d.name} || {dev} || {t} || {sizeof_fmt(d.memory_limit)}")

    # # EXEC SCRIPT
    # ## Dataset prep

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
    lexf = "dict"

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
    lexicon_fn = gallery_prefix + text_sufix + sample_sufix + "." + lexf

    # model names
    dis_model_name = "VVG-Text2Img-CDiscriminator"
    gen_model_name = "VVG-Text2Img-CGenerator"
    gan_model_name = "VVG-Text2Img-CGAN"

    # to continue training after stoping script
    cont_train = True

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
    work_txtf, work_imgf, work_sufix, work_imgt, work_lex = text_fn, imgf_fn, sample_sufix, imgt, lexicon_fn

    print("=== working files ===")
    print("\n", work_txtf, "\n", work_imgf, "\n", work_sufix, "\n", work_imgt, "\n", valt_fn, "\n", work_lex)


    root_folder = os.getcwd()
    # root_folder = os.path.split(root_folder)[0]
    root_folder = os.path.normpath(root_folder)
    print(root_folder)

    # variable reading
    # dataframe filepath for texttual data
    text_fn_path = os.path.join(root_folder, dataf, trainf, work_txtf)
    print(text_fn_path, os.path.exists(text_fn_path))

    # dataframe filepath for img data
    img_fn_path = os.path.join(root_folder, dataf, trainf, work_imgf)
    print(img_fn_path, os.path.exists(img_fn_path))

    # dictionary filepath for the GAN data
    lex_fn_path = os.path.join(root_folder, dataf, trainf, work_lex)
    print(lex_fn_path, os.path.exists(lex_fn_path))

    # dataframe filepath for GAN data
    val_fn_path = os.path.join(root_folder, dataf, testf, valt_fn)
    print(val_fn_path, os.path.exists(val_fn_path))

    # filepath for the models
    model_fn_path = os.path.join(root_folder, dataf, modelf)
    print(model_fn_path, os.path.exists(model_fn_path))

    # filepath for the reports
    report_fn_path = os.path.join(root_folder, dataf, reportf)
    print(report_fn_path, os.path.exists(report_fn_path))


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


    idx_cols = list()

    for tcol in text_cols:
        if tcol in img_cols:
            idx_cols.append(tcol)
    print(idx_cols)

    source_df = pd.merge(text_df, img_df, how="inner", on=idx_cols)


    # checking everything is allrigth
    img_df = None
    text_df = None
    source_df.info()
    source_df = source_df.set_index("ID")

    # reading images from folder and loading images into df
    # working variables
    src_col = work_imgt + "_img"
    tgt_col = work_imgt + "_img" + "_data"
    work_shape = work_imgt + "_shape"
    scale = 16 # !!! 50->400pix, 64->512pix, 32->256pix 16->128pix
    print(src_col, tgt_col)
    source_df = get_images(root_folder, source_df, src_col, tgt_col, scale)


    # update image shape
    source_df = update_shape(source_df, tgt_col, work_shape)


    # searching the biggest shape in the image files
    print(work_shape)
    shape_data = source_df[work_shape]
    max_shape = get_mshape(shape_data, work_imgt)
    print(max_shape)


    # padding training data according to max shape of the images in gallery
    pad_prefix = "pad_"
    conv_prefix = "cnn_"
    src_col = work_imgt + "_img" + "_data"
    tgt_col = pad_prefix + conv_prefix + src_col

    print(src_col, tgt_col, work_imgt)
    source_df = padding_images(source_df, src_col, tgt_col, max_shape, work_imgt)


    # reading images from folder and stadarizing images into df
    # working variables
    print("standarizing regular images...")
    src_col = work_imgt + "_img" + "_data"
    tgt_col = "std_" + src_col

    # source_df = standarize_images(source_df, src_col, tgt_col)
    print("standarizing padded images...")
    src_col = pad_prefix + conv_prefix + work_imgt + "_img" + "_data"
    tgt_col = "std_" + src_col
    print(src_col, tgt_col)

    # std_opt = "std"
    std_opt = "ctr"
    source_df = standarize_images(source_df, src_col, tgt_col, work_imgt, std_opt)


    # shuffle the DataFrame rows
    source_df.info()

    # cleaning memory
    gc.collect()
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


    too_disperse = get_disperse_categories(source_df, keep_columns, 2, "std_cat_", "std_pad_cnn_rgb_img_data")
    print(too_disperse)


    # creating the training dataframe
    keep_columns = remove_disperse_categories(keep_columns, too_disperse)
    # keep_columns.remove("ID")
    print("------ Interesting columns ------")
    print(keep_columns)


    # saving idtfd encoding to translate bow
    tfidf_tokens = source_df["tfidf_tokens"]


    # creating the training dataframe
    train_df = pd.DataFrame(source_df, columns=keep_columns)


    # shuffling the stuff
    train_df = train_df.sample(frac = 1)
    source_df = None
    df_columns = list(train_df.columns)


    train_df.info()


    # getting the column with the relevant data to train
    pad_regex = u"^std_pad_"
    padimg_col = get_keeper_cols(df_columns, pad_regex)
    padimg_col = padimg_col[0]
    print("Padded image column in dataframe: ", str(padimg_col))


    # getting the column with the relevant data to train
    dvec_regex = u"^std_dvec"
    dvector_col = get_keeper_cols(df_columns, dvec_regex)
    dvector_col = dvector_col[0]
    print("Dense vector column in dataframe: ", str(dvector_col))


    # fix column data type
    work_corpus = train_df[dvector_col]
    work_corpus = format_dvector(work_corpus)


    # changing type in dataframe
    train_df[dvector_col] = work_corpus
    work_corpus = None


    # padding training data according to max length of text corpus
    pad_prefix = "pad_"
    recurrent_prefix = "lstm_"

    train_df = padding_corpus(train_df, dvector_col, pad_prefix)


    regular_img_col = "std_" + work_imgt + "_img" + "_data"
    padded_img_col = "std_" + pad_prefix + conv_prefix + work_imgt + "_img" + "_data"
    padded_col_dvector = pad_prefix + dvector_col


    # getting the columns with the relevant labels to predict
    print(keep_columns)
    cat_regex = u"^std_cat_"
    labels_cols = get_keeper_cols(keep_columns, cat_regex)
    print("Classifier trainable labels in dataframe: ", str(labels_cols))

    # updating dataframe with hot/concatenated categories
    tcat_label_col = "std_cat_labels"
    print("categories heat column:", tcat_label_col)
    train_df = heat_categories(train_df, labels_cols, tcat_label_col)


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


    text_lstm_col = padded_col_dvector
    print(text_lstm_col)


    working_img_col = padded_img_col
    # working_img_col = regular_img_col
    print(working_img_col)


    train_df.info()


    gc.collect()


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


    print(type(X_img[0]))
    print(type(X_img[0][0]))
    print(X_img[1].shape)


    if len(X_img.shape) == 3:
        X_img = X_img[:,:,:,np.newaxis]


    # y = train_df[working_img_col]
    # y = np.expand_dims(y, axis=0)
    y_labels = np.asarray([np.asarray(j, dtype="object") for j in train_df[tcat_label_col]], dtype="object")
    print("y shape", y_labels.shape)


    y = np.ones((y_labels.shape[0],1)).astype("float32")
    print("y shape", y.shape)


    print("y classification category")
    print(type(y[0]))
    print(type(y[0][0]))
    print(y[1].shape)

    print("y labels category")
    print(type(y_labels[0]))
    print(type(y_labels[0][0]))
    print(y_labels[1].shape)


    # creating Train/Test sample
    # getting the X, y to train, as is autoencoder both are the same
    X_txt = np.asarray([np.asarray(i, dtype="object") for i in train_df[text_lstm_col]], dtype="object")
    # X = np.array(train_df[text_lstm_col]).astype("object")
    # X = train_df[text_lstm_col]
    print("final X_lstm shape", X_txt.shape)


    print(type(X_txt[0]))
    print(type(X_txt[0][0]))
    print(X_txt[1].shape)


    # timestep is the memory of what i read, this is the longest sentence I can remember in the short term
    # neet to look for the best option, in small the max is 15
    timesteps = 15

    # features is the max length in the corpus, after padding!!!!
    features = X_txt[0].shape[0]
    print(timesteps, features)


    X_txt = format_corpus(X_txt, timesteps, features)


    print(X_txt.shape)


    diff_txt = y.shape[0] - X_txt.shape[0]
    print(diff_txt)


    Xa = X_txt[-diff_txt:]
    X_txt = np.append(X_txt, Xa, axis=0)
    print(X_txt.shape)
    Xa = None


    print(X_txt.shape)
    print(X_img.shape)
    print(y.shape)
    print(y_labels.shape)


    print(X_txt[0].shape)
    print(X_img[0].shape)
    print(y[0].shape)
    print(y_labels[0].shape)
    txt_og_shape = X_txt[0].shape
    img_og_shape = X_img[0].shape
    cat_og_shape = y[0].shape
    lab_og_shape = y_labels[0].shape


    # Xt = X_txt # np.array(X).astype("object")
    # Xi = X_img
    # yt = y # np.array(y).astype("object")
    # # ya = y[0:timesteps]
    # train_df = None
    X_img = X_img.astype("float32")
    X_txt = X_txt.astype("float32")
    y = y.astype("float32")
    y_labels = y_labels.astype("float32")

    print(type(X_img[0]), type(X_txt[0]), type(y_labels[0]), type(y[0]))
    gc.collect()

    # ## ML Models Configuration
    # ### GAN-img definition
    REF_KERNEL_SIZE = (5,5) # (4,4) # (7,7) # (5,5) # (4,4)
    REF_FILTERS = 128*8 #*8 # *16 -> 256pix, *8->128, 
    REF_LSTM = 400

    # slow opti functions
    MIN_DLR = 0.000005
    MIN_GLR = 0.00002

    # fast opti functions
    MAX_DLR = 0.00005
    MAX_GLR = 0.0002

    # middle opti functions
    MID_DLR = 0.000008 # 0.000003
    MID_GLR = 0.00005 # 0.0001

    # working opti functions
    DIS_OPTI_REF = Adam(learning_rate=MAX_DLR, beta_1=0.50) # oficial learning_Rate=0.000050, beta_1= 0.50
    GEN_OPTI_REF = Adam(learning_rate=MAX_GLR, beta_1=0.50) # oficial learning_rate0.000200, beta_1=0.50

    # loss and activation function
    LOSS_REF = "binary_crossentropy"
    ACC_REF = ["accuracy"]
    CNN_ACT_REF = LeakyReLU(alpha=0.2)
    LSTM_ACT_REF = LeakyReLU(alpha=0.2)
    DENSE_ACT_REF = LeakyReLU(alpha=0.2)

    # img -> ImgMultiCDisOut
    # txt -> TxtMultiCDisOut
    MULTI_DIS_LOSS_REF = {"ImgMultiCDisOut":"binary_crossentropy", 
                            "TxtMultiCDisOut":"binary_crossentropy",
                            }
    MULTI_DIS_ACC_REF = {"ImgMultiCDisOut":"accuracy",
                            "TxtMultiCDisOut":"accuracy",
                            }
    #  OPT-1
    MULTI_DIS_WEIGHTS_REF = {"ImgMultiCDisOut":1.0,
                            "TxtMultiCDisOut":1.0,
                            }

    # OPT-2
    # MULTI_DIS_WEIGHTS_REF = {"ImgMultiCDisOut":0.6,
    #                         "TxtMultiCDisOut":0.4,
    #                         }

    # img -> MultiCGAN_ImgPlusTxt_Discriminator
    # txt -> MultiCGAN_ImgPlusTxt_Discriminator_1
    MULTI_GEN_LOSS_REF = {"MultiCGAN_ImgPlusTxt_Discriminator":"binary_crossentropy",
                            "MultiCGAN_ImgPlusTxt_Discriminator_1":"binary_crossentropy",
                            }
    MULTI_GEN_ACC_REF = {"MultiCGAN_ImgPlusTxt_Discriminator":"accuracy", 
                            "MultiCGAN_ImgPlusTxt_Discriminator_1":"accuracy",
                            }
    # OPT-1
    MULTI_GEN_WEIGHTS_REF = {"MultiCGAN_ImgPlusTxt_Discriminator":1.0,
                                "MultiCGAN_ImgPlusTxt_Discriminator_1":1.0,
                            }

    # OPT-2
    # MULTI_GEN_WEIGHTS_REF = {"MultiCGAN_ImgPlusTxt_Discriminator":0.6,
    #                             "MultiCGAN_ImgPlusTxt_Discriminator_1":0.4,
    #                         }

    # common variables for the models
    # input common vars
    input_filters = REF_FILTERS
    input_kernel_size = REF_KERNEL_SIZE
    input_stride = (2,2)
    input_padding = "same"
    input_lstm_neurons = REF_LSTM
    mask_value = 0.0
    input_return_sequences = True

    # latent and conditional label common vars
    # def of the latent space size for the input
    latent_features = 8 # 5 # model_cfg.get("latent_features")
    latent_filters = REF_FILTERS # 128 # model_cfg.get("latent_filters")
    latent_lstm_reshape = X_txt[0].shape
    memory_shape = X_txt[0].shape
    memory = memory_shape[0]
    max_features = memory_shape[1]
    labels_neurons = timesteps*X_txt.shape[2]
    latent_img_size = 8*8*REF_FILTERS # 50*50*8 # 32*32*3, # 5*5*128 #
    latent_img_shape = (8,8,REF_FILTERS) # (50,50,8) # (32,32,3), # (5,5,128) # 
    labels_img_neurons =  8*8*REF_FILTERS # 50*50*3
    labels_filters = REF_FILTERS
    labels_kernel_size = REF_KERNEL_SIZE
    labels_stride = (2,2)
    labels_reshape = (8,8,REF_FILTERS)
    labels_lstm_neurons = REF_LSTM
    labels_return_sequences = True
    labels_txt_reshape = X_txt[0].shape

    # hidden common vars
    lstm_neurons = REF_LSTM
    filters = REF_FILTERS
    kernel_size = REF_KERNEL_SIZE
    stride = (2,2)
    padding = "same"
    gen_dropout_rate = 0.25
    mid_txt_gen_neurons = X_txt.shape[1]*X_txt.shape[2]
    hidden_return_sequences = True
    dis_dropout_rate = 0.25
    mid_dis_neurons = 2*2*REF_FILTERS # 50*50*2 # 32*32*3,
    dense_cls_activation = DENSE_ACT_REF # "softmax"

    # output common vars
    output_neurons = X_txt.shape[2]
    output_txt_shape = X_txt[0].shape
    output_gen_lyr_activation = "tanh" #"softmax" #"tanh"
    output_gen_txt_activation = "softmax" #"tanh"
    output_return_sequences = True
    output_filters = X_img[0].shape[2]
    output_kernel_size = (3,3)
    output_stride = (1,1)
    output_img_shape = X_img[0].shape
    output_dis_neurons = 1
    output_dis_lyr_activation = "sigmoid" # "softmax"


    # img generator config
    img_gen_cfg = {
        "gen_model_name": "GAN_img_Generator",
        "latent_features": latent_features,
        "latent_filters": latent_filters,
        "mask_value": mask_value,
        "return_sequences": hidden_return_sequences,
        "lstm_neurons": lstm_neurons,
        "latent_img_size": latent_img_size,
        "input_lyr_activation": CNN_ACT_REF,
        "latent_img_shape": latent_img_shape,
        "filters": filters, 
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "hidden_lyr_activation": CNN_ACT_REF,
        "gen_dropout_rate": gen_dropout_rate,
        "output_filters": output_filters,
        "output_kernel_size": output_kernel_size,
        "output_stride": output_stride,
        "output_padding": padding,
        "output_shape": output_img_shape,
        "output_lyr_activation": output_gen_lyr_activation,
        }

    print("GAN-img Generator Config:\n", img_gen_cfg)


    # img discriminator config
    img_dis_cfg = {
        "dis_model_name": "GAN_img_Discriminator",
        "input_lyr_activation": CNN_ACT_REF,
        "input_filters": input_filters,
        "input_kernel_size": input_kernel_size,
        "input_stride": input_stride,
        "input_padding": input_padding,
        "filters": filters,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "hidden_lyr_activation": CNN_ACT_REF,
        "dis_dropout_rate": dis_dropout_rate,
        "mid_dis_neurons": mid_dis_neurons,
        "dense_cls_activation": dense_cls_activation,
        "output_dis_neurons": output_dis_neurons,
        "output_lyr_activation": output_dis_lyr_activation,
        "loss": LOSS_REF,
        "optimizer": DIS_OPTI_REF,
        "metrics": ACC_REF,
        }

    print("GAN-img Discriminator Config:\n", img_dis_cfg)


    # img GAN config
    gan_cfg = {
        "gan_model_name": "GAN_img",
        "loss": LOSS_REF,
        "optimizer": GEN_OPTI_REF,
        "metrics": ACC_REF,
        }

    print("GAN-img Config:\n", gan_cfg)

    # ### GAN-txt definition

    # txt generator config
    txt_gen_cfg = {
        # "gen_model_name": "GAN_txt_Generator",
        "mask_value": mask_value,
        "input_return_sequences": input_return_sequences,
        "input_lstm_neurons": input_lstm_neurons,
        "input_lyr_activation": LSTM_ACT_REF,
        "latent_txt_size": mid_txt_gen_neurons,
        "lstm_neurons": lstm_neurons,
        "hidden_lyr_activation": LSTM_ACT_REF,
        "hidden_return_sequences": hidden_return_sequences,
        "gen_dropout_rate": gen_dropout_rate,
        "latent_txt_shape": latent_lstm_reshape,
        "memory_shape": memory_shape,
        "output_neurons": output_neurons,
        "output_shape": output_txt_shape,
        "output_txt_activation": output_gen_txt_activation,
        "output_return_sequences": output_return_sequences,
        }

    print("GAN-txt Generator Config:\n", txt_gen_cfg)


    # txt discriminator config
    txt_dis_cfg = {
        # "dis_model_name": "GAN-txt Discriminator",
        "mask_value": mask_value,
        "input_return_sequences": input_return_sequences,
        "input_lstm_neurons": input_lstm_neurons,
        "input_lyr_activation": LSTM_ACT_REF,
        "lstm_neurons": lstm_neurons,
        "hidden_return_sequences": hidden_return_sequences,
        "hidden_lyr_activation": LSTM_ACT_REF,
        "memory_shape": memory_shape,
        "dis_dropout_rate": dis_dropout_rate,
        "latent_txt_size": mid_txt_gen_neurons,
        "mid_dis_neurons": mid_txt_gen_neurons,
        "dense_cls_activation": dense_cls_activation,
        "output_dis_neurons": output_dis_neurons,
        "output_lyr_activation": output_dis_lyr_activation,
        "loss": LOSS_REF,
        "optimizer": DIS_OPTI_REF,
        "metrics": ACC_REF,
        }

    print("GAN-txt Discriminator Config:\n", txt_dis_cfg)

    # ### CGAN-img definition

    img_cgen_cfg = {
        "gen_model_name": "CGAN_img_Generator",
        "latent_features": latent_features,
        "latent_filters": latent_filters,
        "memory": memory,
        "features": max_features,
        "mask_value": mask_value,
        "latent_img_size": latent_img_size,
        "input_lyr_activation": CNN_ACT_REF,
        "latent_img_shape": latent_img_shape,
        "filters": filters, 
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "hidden_lyr_activation": CNN_ACT_REF,
        "gen_dropout_rate": gen_dropout_rate,
        "output_filters": output_filters,
        "output_kernel_size": output_kernel_size,
        "output_stride": output_stride,
        "output_padding": padding,
        "output_shape": output_img_shape,
        "output_lyr_activation": output_gen_lyr_activation,
        "labels_neurons": labels_neurons,
        "labels_lyr_activation": DENSE_ACT_REF,
        }

    print("CGAN-img Generator Config:\n", img_cgen_cfg)


    img_cdis_cfg = {
        "dis_model_name": "CGAN_img_Discriminator",
        "input_lyr_activation": CNN_ACT_REF,
        "input_filters": input_filters,
        "latent_img_size": latent_img_size,
        "latent_img_shape": latent_img_shape,
        "input_kernel_size": input_kernel_size,
        "input_stride": input_stride,
        "input_padding": padding,
        "filters": filters,
        "kernel_size": kernel_size,
        "stride": stride,
        "padding": padding,
        "hidden_lyr_activation": CNN_ACT_REF,
        "dis_dropout_rate": dis_dropout_rate,
        "mid_dis_neurons":mid_dis_neurons,
        "dense_cls_activation": dense_cls_activation,
        "output_dis_neurons": output_dis_neurons,
        "output_lyr_activation": output_dis_lyr_activation,
        "labels_lyr_activation": DENSE_ACT_REF,
        "timesteps": memory,
        "max_features": max_features,
        "labels_neurons": labels_img_neurons,
        "labels_filters": labels_filters,
        "labels_kernel_size": labels_kernel_size,
        "labels_stride": labels_stride,
        "labels_reshape": labels_reshape,
        "loss": LOSS_REF,
        "optimizer": DIS_OPTI_REF,
        "metrics": ACC_REF,
        }

    print("CGAN-img Generator Config:\n", img_cdis_cfg)


    # txt GAN config
    img_cgan_cfg = {
        "gan_model_name": "CGAN_img",
        "loss": LOSS_REF,
        "optimizer": GEN_OPTI_REF,
        "metrics": ACC_REF,
        }

    print("CGAN-img Config:\n", img_cgan_cfg)

    # ### Multi CGAN definition (txt+img)

    multi_cgen_cfg = dict()
    multi_cgen_cfg.update(img_cgen_cfg)
    multi_cgen_cfg.update(txt_gen_cfg)

    mcgen_cfg_update = {
        "gen_model_name": "MultiCGAN_ImgPlusTxt_Generator",
        }

    multi_cgen_cfg.update(mcgen_cfg_update)

    print("Multi CGen-txt2img Config:\n", multi_cgen_cfg)


    multi_cdis_cfg = dict()
    multi_cdis_cfg.update(img_cdis_cfg)
    multi_cdis_cfg.update(txt_dis_cfg)

    mcdis_cfg_update = {
        "dis_model_name": "MultiCGAN_ImgPlusTxt_Discriminator",
        "labels_lstm_neurons": labels_lstm_neurons,
        "labels_return_sequences": labels_return_sequences,
        "labels_img_reshape": labels_reshape,
        "labels_txt_reshape": labels_txt_reshape,
        "output_txt_activation": output_gen_txt_activation,
        "loss": MULTI_DIS_LOSS_REF,
        "optimizer": DIS_OPTI_REF,
        "metrics":MULTI_DIS_ACC_REF,
        "loss_weights": MULTI_DIS_WEIGHTS_REF,
        }

    multi_cdis_cfg.update(mcdis_cfg_update)

    print("Multi CDis-txt2img Config:\n", multi_cdis_cfg)


    # txt2img CGAN config
    multi_cgan_cfg = {
        "gan_model_name": "Multi_CGAN_ImgPlusTxt",
        "loss": MULTI_GEN_LOSS_REF,
        "optimizer": GEN_OPTI_REF,
        "metrics": MULTI_GEN_ACC_REF,
        "loss_weights": MULTI_GEN_WEIGHTS_REF,
        }

    print("Multi CGAN-txt2img Config:\n", multi_cgan_cfg)

    # ## ML Model Creation
    # ### GAN img definition

    # latent shape
    latent_dims = 128
    print(latent_dims)
    # latent_shape = (int(X_img[0].shape[0]/4), int(X_img[0].shape[1]/4), 3)
    # latent_shape = (100, 100)


    gen_model = create_img_generator(latent_dims, img_gen_cfg)
    print("GAN-img Generator Definition")
    # dis_model = Sequential(slim_dis_layers)
    # gen_model.model_name = "GAN-img Generator"

    # DONT compile model
    # cdis_model.trainable = False
    gen_model.summary()


    img_shape = X_img[0].shape
    print(img_shape)

    # img_shape = (100,100,3)
    dis_model = create_img_discriminator(img_shape, img_dis_cfg)
    print("GAN-img Discriminator Definition")
    # dis_model = Sequential(slim_dis_layers)
    # dis_model.model_name = "GAN-img Discriminator"

    # compile model
    dis_model.compile(loss=img_dis_cfg["loss"], 
                        optimizer=img_dis_cfg["optimizer"], 
                        metrics=img_dis_cfg["metrics"])

    # cdis_model.trainable = False
    dis_model.summary()


    print("GAN-img Model definition")
    gan_model = create_img_gan(gen_model, dis_model, gan_cfg)
    # gan_model.model_name = "GAN-img"
    gan_model.summary()


    # saving model topology into png files
    print(timestamp)
    export_model(gen_model, model_fn_path, img_gen_cfg.get("gen_model_name"), timestamp)
    export_model(dis_model, model_fn_path, img_dis_cfg.get("dis_model_name"), timestamp)
    export_model(gan_model, model_fn_path, gan_cfg.get("gan_model_name"), timestamp)

    # ### GAN txt definition

    # gen_txt_model = create_txt_generator(latent_dims, txt_gen_cfg)
    # print("GAN-txt Generator Definition")
    # # dis_model = Sequential(slim_dis_layers)
    # gen_txt_model.model_name = "GAN-txt Generator"

    # # DONT compile model
    # # cdis_model.trainable = False
    # gen_txt_model.summary()


    txt_shape = X_txt[0].shape
    print(txt_shape)
    # dis_txt_model = create_txt_discriminator(txt_shape, txt_dis_cfg)
    # print("GAN-txt Discriminator Definition")
    # # dis_model = Sequential(slim_dis_layers)
    # dis_txt_model.model_name = "GAN-txt Discriminator"

    # # compile model
    # dis_txt_model.compile(loss=txt_dis_cfg["loss"], 
    #                     optimizer=txt_dis_cfg["optimizer"], 
    #                     metrics=txt_dis_cfg["metrics"])

    # # cdis_model.trainable = False
    # dis_txt_model.summary()


    print("GAN-txt Model definition")
    # gan_txt_model = create_img_gan(gen_txt_model, dis_txt_model, gan_cfg)
    # gan_txt_model.summary()
    # gan_txt_model.model_name = "GAN-txt"


    # saving model topology into png files
    print(timestamp)
    # export_model(gen_txt_model, model_fn_path, gen_txt_model.model_name, timestamp)
    # export_model(dis_txt_model, model_fn_path, dis_txt_model.model_name, timestamp)
    # export_model(gan_txt_model, model_fn_path, gan_txt_model.model_name, timestamp)

    # ### CGAN definition

    n_labels = y_labels[0].shape[0]
    print(n_labels)
    cgen_img_model = create_img_cgenerator(latent_dims, n_labels, img_cgen_cfg)
    print("CGAN-img Generator Definition")
    # dis_model = Sequential(slim_dis_layers)
    # cgen_img_model.model_name = "CGAN-img Generator"

    # DONT compile model
    # cdis_model.trainable = False
    cgen_img_model.summary()


    img_shape = X_img[0].shape
    print(img_shape)
    cdis_img_model = create_img_cdiscriminator(img_shape, n_labels, img_cdis_cfg)
    print("CGAN-img Discriminator Definition")
    # dis_model = Sequential(slim_dis_layers)
    # cdis_img_model.model_name = "CGAN-img Discriminator"

    # compile model
    cdis_img_model.compile(loss=img_cdis_cfg["loss"], 
                        optimizer=img_cdis_cfg["optimizer"], 
                        metrics=img_cdis_cfg["metrics"])

    # cdis_model.trainable = False
    cdis_img_model.summary()


    print("CGAN-img Model definition")
    cgan_img_model = create_img_cgan(cgen_img_model, cdis_img_model, img_cgan_cfg)
    cgan_img_model.summary()
    # cgan_img_model.model_name = "CGAN-img"


    # saving model topology into png files
    print(timestamp)
    export_model(cgen_img_model, model_fn_path, img_cgen_cfg.get("gen_model_name"), timestamp)
    export_model(cdis_img_model, model_fn_path, img_cdis_cfg.get("dis_model_name"), timestamp)
    export_model(cgan_img_model, model_fn_path, img_cgan_cfg.get("gan_model_name"), timestamp)

    # ### Multi CGAN-txt&img

    multi_cgen_model = create_multi_cgenerator(latent_dims, img_shape, txt_shape, n_labels, multi_cgen_cfg)
    print("Multi CGAN-txt2img Generator Definition")
    # dis_model = Sequential(slim_dis_layers)
    # multi_cgen_model.model_name = "Multi CGAN-txt&img Generator"

    # DONT compile model
    # cdis_model.trainable = False
    multi_cgen_model.summary()


    print(txt_shape)
    multi_cdis_model = create_multi_cdiscriminator(img_shape, txt_shape, n_labels, multi_cdis_cfg)
    print("Multi CGAN-txt2img Discriminator Definition")
    # multi_cdis_model.model_name = "Multi CGAN-txt&img Discriminator"
    # compile model

    multi_cdis_model.compile(loss=multi_cdis_cfg["loss"], 
                            optimizer=multi_cdis_cfg["optimizer"], 
                            metrics=multi_cdis_cfg["metrics"],
                            loss_weights=multi_cdis_cfg["loss_weights"])

    # compile model
    multi_cdis_model.summary()


    print("Multi CGAN-txt2img Model definition")
    multi_cgan_model = create_multi_cgan(multi_cgen_model, multi_cdis_model, multi_cgan_cfg)
    multi_cgan_model.summary()
    # multi_cgan_model.model_name = "Multi CGAN-txt&img"


    # saving model topology into png files
    print(timestamp)
    export_model(multi_cgen_model, model_fn_path, multi_cgen_cfg.get("gen_model_name"), timestamp)
    export_model(multi_cdis_model, model_fn_path, multi_cdis_cfg.get("dis_model_name"), timestamp)
    export_model(multi_cgan_model, model_fn_path, multi_cgan_cfg.get("gan_model_name"), timestamp)


    print("-Images:", X_img.shape, "\n-Text:", X_txt.shape, "\n-Real/Fake:", y.shape, "\n-txt&img Labels:", y_labels.shape)

    # training and batch size
    gan_train_cfg = {
        "max_epochs": 100, #1000*100*3,
        # "max_epochs": 1000*100*3, # real 1-2
        "latent_dims": latent_dims,
        # "max_epochs": ini_config.get("Training", "MaxEpochs"),
        "trained_epochs": 0,
        "batch_size": 32*1,
        # "batch_size": 32*2, # real 1-2
        "synth_batch": 1,
        "balance_batch": False,
        "gen_sample_size": 3,
        "models_fn_path": model_fn_path,
        "report_fn_path": report_fn_path,
        "learning_history": None,
        "gen_model_name": multi_cgen_cfg.get("gen_model_name"),
        "dis_model_name": multi_cdis_cfg.get("dis_model_name"),
        "gan_model_name": multi_cgan_cfg.get("gan_model_name"),
        # "dis_model_name": img_cdis_cfg.get("dis_model_name"),
        # "gen_model_name": img_cgen_cfg.get("gen_model_name"),
        # "gan_model_name": img_cgan_cfg.get("gan_model_name"), 
        # "gen_model_name": img_gen_cfg.get("gen_model_name"),
        # "dis_model_name": img_dis_cfg.get("dis_model_name"),
        # "gan_model_name": gan_cfg.get("gan_model_name"),
        "check_epochs": 10*1,
        "save_epochs": 50*1,
        # "check_epochs": 10*10, # OPT-2
        # "save_epochs": 10*10*5, # OPT-2
        # "check_epochs": 10*10, # OPT-1
        # "save_epochs": 10*10*3, # OPT-1
        "max_save_models": 3,
        # "max_save_models": 12, # OPT-1-2
        "latent_dims": latent_dims, # X_txt[0].shape,
        "trained": False,
        "conditioned": True,
        "dataset_size": X_img.shape[0],
        "img_shape": X_img[0].shape,
        "txt_shape": X_txt[0].shape,
        "label_shape": y_labels[0].shape,
        "cat_shape": y[0].shape,
        # "data_cols": 2,
        # "data_cols": 3,
        "data_cols": 4,
        "bow_lexicon": load_lexicon(lex_fn_path),
        "tfidf_lexicon": format_tfidf_tokens(tfidf_tokens.values),
        }

    print("Model Training Config:\n", gan_train_cfg.keys())

    # gan_data = (X_img, y)
    # gan_data = (X_img, y_labels, y)
    gan_data = (X_img, X_txt, y_labels, y)
    print(X_img.shape, X_txt.shape, y_labels.shape, y.shape)
    print(len(gan_data))

    # traininng with the traditional gan
    # training_model(gen_model, dis_model, gan_model, gan_data, gan_train_cfg)

    # training with the conditional gan with images
    # training_model(cgen_img_model, cdis_img_model, cgan_img_model, gan_data, gan_train_cfg)

    # training with the muti conditional gan with images + text
    training_model(multi_cgen_model, multi_cdis_model, multi_cgan_model, gan_data, gan_train_cfg)

    # if model is pretrained
    ############ LOading pretrained models!!!! #################
    LAST_EPOCH = 50
    # LAST_EPOCH = 4300 # OPT-1-2
    gan_train_cfg["trained"] = True
    gan_train_cfg["trained_epochs"] = LAST_EPOCH
    # print("---- gan_train_cfg ----\n", gan_train_cfg)

    # training with the muti conditional gan with images + text
    training_model(multi_cgen_model, 
                    multi_cdis_model, 
                    multi_cgan_model, 
                    gan_data, 
                    gan_train_cfg)



    # # THE END
