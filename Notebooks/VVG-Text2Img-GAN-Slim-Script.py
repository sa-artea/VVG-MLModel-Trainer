# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-toolsai.jupyter added
import os
try:
	os.chdir(os.path.join(os.getcwd(), '..\\..\..\..\AppData\Local\Temp\fadc45f7-d192-4067-99e6-ba2f3c5bebd3'))
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
from keras.preprocessing.sequence import pad_sequences
from keras.utils.vis_utils import plot_model
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping

# models
from keras.models import Sequential

# shapping layers
from keras.layers import Masking
from keras.layers import Flatten
from keras.layers import Reshape
from keras.layers import Embedding
from keras.layers import Concatenate

# basic layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import TimeDistributed

# data processing layers
from keras.layers import BatchNormalization
from keras.layers import Dropout
from keras.layers import SpatialDropout1D

# recurrent and convolutional layers
from keras.layers import LSTM
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D

# activarion function
from keras.layers import LeakyReLU

# optimization loss functions
from keras.initializers import RandomNormal
from keras.optimizers import SGD # OJO!
from keras.optimizers import RMSprop
from keras.optimizers import Adam # OJO!
from keras.optimizers import Adadelta # OJO!
from keras.optimizers import Adagrad # OJO!

# image augmentation and processing
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator

# ===============================
# developed python libraries
# ===============================


# %%
# GPU config if I have
physical_devices = tf.config.list_physical_devices("GPU")
print(physical_devices)

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
    for i in range(len(X)-lookback-1):
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
# function to augment the images in the dataset and virtualy exapnd the training examples
def augment_images(src_df, src_col, tgt_col, syth_num):

    cols = [list(src_df.columns.values)]
    # print(cols)
    ans = pd.DataFrame()
    other_cols = list(src_df.columns.values)
    other_cols.remove(tgt_col)
    other_cols.remove(src_col)
    # print(other_cols)

    for index, row in src_df.iterrows():
        t_txt = row[src_col]
        t_img = row[tgt_col]
        t_tags = row[other_cols]

        gen_rows = list()
        for i in range(syth_num):

            gen_tags = copy.deepcopy(t_tags)
            gen_img = syth_img(t_img)
            gen_txt = syth_text(t_txt)
            # print(type(gen_tags), type(gen_img)) 
            gen_tags[tgt_col] = gen_img
            gen_tags[src_col] = gen_txt
            gen_rows.append(gen_tags)
            # print(gen_tags) # , type(gen_img)) 
            # [other_cols], row[tgt_col])
        
        ans = ans.append(gen_rows, ignore_index=True)

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
    model.save(fpn)


# function to load the ML model
def load_model(m_path, m_file):

    fpn = os.path.join(m_path, m_file)
    fpn = fpn + ".h5"
    model = keras.models.load_model(fpn)
    return model


# %%
# function to cast dataframe and avoid problems with keras
def cast_batch(X_txt, X_img, y):
    X_txt = np.asarray(X_txt).astype("float32")
    X_img = np.asarray(X_img).astype("float32")
    y = np.asarray(y).astype("float32")
    return X_txt, X_img, y


# %%
# function to select real elements to train the discriminator
def gen_real_samples(X_txt, X_img, y, sample_size, half_batch):

    rand_index = np.random.randint(0, sample_size, size=half_batch)
    Xt_real = X_txt[rand_index]
    Xi_real = X_img[rand_index]
    y_real = y[rand_index]
    # noise = np.random.uniform(0.0, 0.05, size=y_real.shape)
    # y_real = np.subtract(y_real, noise)
    Xt_real, Xi_real, y_real = cast_batch(Xt_real, Xi_real, y_real)

    return Xt_real, Xi_real, y_real


# %%
# function to create fake elements to train the discriminator
def gen_fake_samples(gen_model, txt_shape, half_batch, cat_size):
    # random text
    Xt_fake = gen_latent_txt(txt_shape, half_batch)
    # random generated image from the model
    Xi_fake = gen_model.predict(Xt_fake)
    # marking the images as fake in all accounts
    y_fake = get_fake_negative(half_batch, cat_size)
    # y_fake = np.zeros((half_batch, cat_size), dtype="float32")
    # casting data type
    Xt_fake, Xi_fake, y_fake = cast_batch(Xt_fake, Xi_fake, y_fake)

    return Xt_fake, Xi_fake, y_fake


# %%
# function to create one fake + real samples to train the discriminator
def complete_batch(Xt_real, Xi_real, y_real, Xt_fake, Xi_fake, y_fake):

    # this batch needs txt to create images, the images themselves, and the images labels
    Xt = np.concatenate((Xt_real, Xt_fake), axis=0)
    Xi = np.concatenate((Xi_real, Xi_fake), axis=0)
    y = np.concatenate((y_real, y_fake), axis=0)
    # Xt, Xi, y = cast_batch(Xt, Xi, y)
    
    return Xt, Xi, y


# %%
# function to generate random/latent text for image generator
def gen_latent_txt(txt_shape, txt_samples):

    ans = None
    for i in range(txt_samples):

        # be aware of this!!!!!!!
        noise = np.random.normal(0.0, 1.0, size=txt_shape)
        if ans is None:
            txt = np.expand_dims(noise, axis=0)
            ans = txt
        else:
            img = np.expand_dims(txt, axis=0)
            ans = np.concatenate((ans, txt), axis=0)
    # print(ans.shape)
    # print(ans[0])
    return ans


# %%
# tfunction to smooth the fake positives
def smooth_positive_labels(y):
	return y - 0.3 + (np.random.random(y.shape)*0.5)


# %%
# tfunction to smooth the fake negatives
def smooth_negative_labels(y):
	return y + np.random.random(y.shape)*0.3


# %%
# randomly flip some labels
def noisy_labels(y, p_flip):
	# determine the number of labels to flip
	n_select = int(p_flip * y.shape[0])
	# choose labels to flip
	flip_ix = np.random.choice([i for i in range(y.shape[0])], size=n_select)
	# invert the labels in place
	y[flip_ix] = 1 - y[flip_ix]
	return y


# %%
# generate fake true categories for the generator
def get_fake_cat(batch_size, cat_size):

    sz = (batch_size, cat_size)
    ans = np.ones(sz)
    # smooothing fakes
    ans = smooth_positive_labels(ans)
    ans = ans.astype("float32")
    # ans = np.ones((batch_size, cat_size), dtype="float32")
    return ans


# %%
# generate fake negative categories to train the GAN
def get_fake_negative(batch_size, cat_size):

    sz = (batch_size, cat_size)
    ans = np.zeros(sz)
    ans = smooth_negative_labels(ans)
    ans = ans.astype("float32")
    # ans = np.ones((batch_size, cat_size), dtype="float32")
    return ans


# %%
# generate an expanded bath of images for training with some synthetic ones
def gen_synthetic_images(X_img, img_size, batch_size, synth_size):

    ans = None

    # iterating the images and synth new ones
    for img in X_img:
        gen_img = None

        # creating new ones
        for j in range(synth_size):

            if gen_img is None:
                timg = syth_std_img(img)
                timg = np.expand_dims(timg, axis=0)
                gen_img = timg
            
            else:
                timg = syth_std_img(img)
                timg = np.expand_dims(timg, axis=0)
                gen_img = np.concatenate((gen_img, timg), axis=0)
        
        # adding it to the training batch
        if ans is None:
            ans = gen_img

        else:
            ans = np.concatenate((ans, gen_img), axis=0)

    return ans


# %%
# function to create text similar to the original one with 5% of noise
def syth_text(data, nptc=0.02):

    ans = None
    noise = np.random.normal(0, nptc, data.shape)
    ans = data + noise
    return ans


# %%
# synthetizing a noisy std image from real data
def syth_std_img(data):

    samples = np.expand_dims(data, 0)
    datagen = ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    # datagen = ImageDataGenerator(rotation_range=10, horizontal_flip=True, vertical_flip=True)
    ans = datagen.flow(samples, batch_size=1)
    ans = ans[0].astype("float32")
    ans = np.squeeze(ans, 0)
    return ans


# %%
# function to create new categories with some noise, default 5%
def syth_categories(data, nptc=0.02):

    ans = None
    noise = np.random.normal(0, nptc, data.shape)
    ans = data + noise
    return ans


# %%
# function to artificially span a batch with some noise and alterations by an specific number
def expand_samples(X_txt, X_img, y, synth_batch):

    # creating the exapnded batch response
    Xe_txt, Xe_img, ye = None, None, None

    # iterating in the original batch
    for Xtt, Xit, yt in zip(X_txt, X_img, y):

        # temporal synth minibatch per original image
        synth_Xt, synth_Xi, synth_y = None, None, None

        # synthetizing artificial data for the batch
        for i in range(synth_batch):

            # generating first element
            if (synth_Xt is None) and (synth_Xi is None) and (synth_y is None):
                # gen text
                gen_Xt = syth_text(Xtt)
                gen_Xt = np.expand_dims(gen_Xt, axis=0)
                synth_Xt = gen_Xt

                # gen images
                gen_Xi = syth_std_img(Xit)
                gen_Xi = np.expand_dims(gen_Xi, axis=0)
                synth_Xi = gen_Xi

                # gen labels
                gen_yt = syth_categories(yt)
                gen_yt = np.expand_dims(gen_yt, axis=0)
                synth_y = gen_yt

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

                # gen labels
                gen_yt = syth_categories(yt)
                gen_yt = np.expand_dims(gen_yt, axis=0)
                synth_y = np.concatenate((synth_y, gen_yt), axis=0)
        
        # adding the first part to the training batch
        if (Xe_txt is None) and (Xe_img is None) and (ye is None):
            # adding text
            Xe_txt = synth_Xt
            # adding images
            Xe_img = synth_Xi
            # adding categories
            ye = synth_y

        # adding the rest of the batch
        else:
            # adding text
            Xe_txt = np.concatenate((Xe_txt, synth_Xt), axis=0)
            # adding images
            Xe_img = np.concatenate((Xe_img, synth_Xi), axis=0)
            # adding categories
            ye = np.concatenate((ye, synth_y), axis=0)

    Xe_txt, Xe_img, ye = cast_batch(Xe_txt, Xe_img, ye)

    return Xe_txt, Xe_img, ye


# %%
def drift_labels(Xt_real, Xi_real, y_real, Xt_fake, Xi_fake, y_fake, batch_size, drift_pct):

    # setting the size for the drift labels
    drift_size = int(math.ceil(drift_pct*batch_size))
    # random index for drift elements!!!
    rand_drifters = np.random.choice(batch_size, size=drift_size, replace=False)
    # print("batch size", batch_size, "\nrandom choise to change", drift_size, "\n", rand_drifters)

    for drift in rand_drifters:

        # copying temporal real data
        Xt_drift = copy.deepcopy(Xt_real[drift])
        Xi_drift = copy.deepcopy(Xi_real[drift])
        y_drift = copy.deepcopy(y_real[drift])
        # print("OG real y:", y_drift)
        # print("OG fake y:", y_fake[drift])
        
        # replacing real with fakes
        Xt_real[drift] = copy.deepcopy(Xt_fake[drift])
        Xi_real[drift] = copy.deepcopy(Xi_fake[drift])
        y_real[drift] = copy.deepcopy(y_fake[drift])
        # print("New real y:", y_real[drift])

        # updating fakes with temporal original
        Xt_fake[drift] = Xt_drift
        Xi_fake[drift] = Xi_drift
        y_fake[drift] = y_drift
        # print("New fake y:", y_fake[drift])

    return Xt_real, Xi_real, y_real, Xt_fake, Xi_fake, y_fake


# %%
# functioon to log the training results
def test_model(epoch, gen_model, dis_model, X_txt, X_img, y, txt_shape, cat_shape, img_size, half_batch, report_fn_path, synth_batch):
    # select real txt2img for discrimintator
    Xt_real, Xi_real, y_real = gen_real_samples(X_txt, X_img, y, img_size, half_batch)
    Xt_real, Xi_real, y_real = expand_samples(Xt_real, Xi_real, y_real, synth_batch)

    # create false txt for txt2img for generator
    Xt_fake, Xi_fake, y_fake = gen_fake_samples(gen_model, txt_shape, half_batch, cat_shape[0])
    Xt_fake, Xi_fake, y_fake = expand_samples(Xt_fake, Xi_fake, y_fake, synth_batch)

    plot_gen_images(Xi_fake, epoch, report_fn_path, 3)

    real_batch = int((half_batch*synth_batch)/2)

    # drift labels to confuse the model
    Xt_real, Xi_real, y_real, Xt_fake, Xi_fake, y_fake = drift_labels(Xt_real, Xi_real, y_real, 
                                                                        Xt_fake, Xi_fake, y_fake, 
                                                                        real_batch, 0.05)
                                                                        
    # evaluate model
    testl_real = dis_model.evaluate(Xi_real, y_real, verbose=0)
    testl_fake = dis_model.evaluate(Xi_fake, y_fake, verbose=0)

    # summarize discriminator performance
    print("Batch Size %d -> Samples: Fake: %d & Real: %d" % (half_batch*synth_batch, real_batch, real_batch))
    print(">>> Test Fake -> Acc: %.3f || Loss: %.3f" % (testl_fake[1], testl_fake[0]))
    print(">>> Test Real -> Acc: %.3f || Loss: %.3f" % (testl_real[1], testl_real[0]))


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
        # gimg = destd_img(gimg, 0, 255, "std")
        # gimg*255
        # gimg = np.asarray(gimg).astype("uint8")

        # turn off axis
        plt.axis("off")
        plt.imshow(gimg) #, interpolation="nearest")

    # config axis
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)
    # plot leyend
    fig.suptitle("GENERATED IMAGES", fontsize=50)
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

    # acc_
    ax2.plot(disr_hist[:,0], "royalblue", label="Acc: R-Dis")
    ax2.plot(disf_hist[:,0], "crimson", label="Acc: F-Dis")
    ax2.plot(gan_hist[:,0], "blueviolet", label="Acc: GAN/Gem")

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
    headers = ["dis_loss_real", "dis_acc_real", "dis_loss_fake", "dis_acc_fake", "gen_gan_loss","gen_gan_acc"]

    # formating fake discriminator train data
    drhl = disr_hist[:,1]# .flatten()
    # drhl = drhl.tolist()
    drha = disr_hist[:,0]# .flatten()
    # drha = drha.tolist()

    # formating real discrimintator train data
    dfhl = disf_hist[:,1]# .flatten()
    # dfhl = dfhl.tolist()
    dfha = disf_hist[:,0]# .flatten()
    # dfha = dfha.tolist()

    # formating gan/gen train data
    gghl = gan_hist[:,1]# .flatten()
    # gghl = gghl.tolist()
    ggha = gan_hist[:,0]#.flatten()
    # ggha = ggha.tolist()

    # adding all formatted data into list
    data = np.column_stack((drhl, drha, dfhl, dfha, gghl, ggha))
    # data = pd.DataFrame(values, columns=headers)
    return data, headers


# %%
# function to write data in csv file
def write_metrics(data, headers, report_fp_name, filename):

    # print(report_fp_name, filename)
    fpn = filename + "-train-history.csv"
    fpn = os.path.join(report_fp_name, fpn)

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
def save_metrics(disr_history, disf_history, gan_history, report_fp_name, filename):

    data, headers = format_metrics(disr_history, disf_history, gan_history)
    write_metrics(data, headers, report_fp_name, filename)


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
# special function to train the GAN
# https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
def train(gen_model, dis_model, gan_model, X_img, X_txt, y, epochs, batch_size, save_intervas, fn_config):

    # sample shape
    txt_shape = X_txt[0].shape
    img_shape = X_img[0].shape
    cat_shape = y[0].shape

    # sample size
    txt_size = X_txt.shape[0]
    img_size = X_img.shape[0]
    cat_size = y.shape[0]
    synth_batch = 1 # OJO!
    n = 3

    # model IO configuration
    model_fn_path = fn_config[0]
    report_fn_path = fn_config[1]
    dis_model_name = fn_config[2]
    gen_model_name = fn_config[3]
    gan_model_name = fn_config[4]

    # fake/real batch division
    half_batch = int(batch_size/2)
    batch_per_epoch = int(txt_size/batch_size)
    real_batch = int((batch_size*synth_batch)/2)
    # batch_per_epoch = int((txt_size*synth_batch)/batch_size)

	# prepare lists for storing stats each epoch
    # disf_hist, disr_hist, gen_hist, gan_hist = list(), list(), list(), list()
    disf_hist, disr_hist, gan_hist = list(), list(), list()
    train_time = None
    # iterating in training epochs:
    for ep in range(epochs+1):
        # epoch logs
        # ep_disf_hist, ep_disr_hist, ep_gen_hist, ep_gan_hist = list(), list(), list(), list()
        ep_disf_hist, ep_disr_hist, ep_gan_hist = list(), list(), list()
        train_time = datetime.datetime.now()

        # iterating over training batchs
        for batch in range(batch_per_epoch):

            # select real txt2img for discrimintator
            Xt_real, Xi_real, y_real = gen_real_samples(X_txt, X_img, y, img_size, half_batch)
            # expand the training sample for the discriminator
            Xt_real, Xi_real, y_real = expand_samples(Xt_real, Xi_real, y_real, synth_batch)
            print(y_real)
            # create false txt for txt2img for generator
            Xt_fake, Xi_fake, y_fake = gen_fake_samples(gen_model, txt_shape, half_batch, cat_shape[0])
            # expand the training sample for the discriminator
            Xt_fake, Xi_fake, y_fake = expand_samples(Xt_fake, Xi_fake, y_fake, synth_batch)

            # print(Xt_real.shape, Xi_real.shape, y_real.shape)
            # print(Xt_fake.shape, Xi_fake.shape, y_fake.shape)
            # drift labels to confuse the model
            Xt_real, Xi_real, y_real, Xt_fake, Xi_fake, y_fake = drift_labels(Xt_real, Xi_real, y_real, 
                                                                                Xt_fake, Xi_fake, y_fake,
                                                                                real_batch, 0.05)
    
            # train for real samples batch
            dhr = dis_model.train_on_batch(Xi_real, y_real)
            dhr = dis_model.train_on_batch(Xi_real, y_real)
            dhr = dis_model.train_on_batch(Xi_real, y_real)

            # train for fake samples batch
            dhf = dis_model.train_on_batch(Xi_fake, y_fake)

            # prepare noisy text of latent space as input for the generator
            Xt_gen = gen_latent_txt(txt_shape, batch_size)
            # create inverted labels for the fake noisy text
            y_gen = get_fake_cat(batch_size, cat_shape[0])
            # update the generator via the discriminator's error
            gh = gan_model.train_on_batch(Xt_gen, y_gen)
            # print("ojo GAN!", gh)

            ep_disr_hist.append(dhf)
            ep_disf_hist.append(dhr)
            # ep_gen_hist.append(gh)
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
        # gen_hist.append(epoch_avg(ep_gen_hist))
        gan_hist.append(epoch_avg(ep_gan_hist))

		# evaluate the model performance sometimes
        if (ep) % save_intervas == 0:
            print("Epoch:", ep+1, "Saving the training progress...")

            test_model(ep, gen_model, dis_model, X_txt, X_img, y, txt_shape, cat_shape, img_size, half_batch, report_fn_path, synth_batch)
            plot_metrics(disr_hist, disf_hist, gan_hist, report_fn_path, ep)
            save_metrics(disr_hist, disf_hist, gan_hist, report_fn_path, gan_model_name)

		# saving the model sometimes
        if (ep) % int(save_intervas*5) == 0:
            epoch_sufix = "-epoch%3d" % int(ep)
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
dis_model_name = "VVG-Text2Img-S-Discriminator"
gen_model_name = "VVG-Text2Img-S-Generator"
gan_model_name = "VVG-Text2Img-S-GAN"

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

# data augmentation
# source_df = augment_images(source_df, src_col, tgt_col, 6)
# source_df.info()


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
# ## ML Model Definition

# %%
# number of neurons or processing units in LSTM
# the number is because of good practices for NLP
# min 200 max 500, normaly 300 (related to the semantic number of themes)
# 120 for now in this test
lstm_units = 400
print("Generator LSMT neurons:", lstm_units)

# timestep is 1 because you read a word at a time
memory = timesteps
print("Generator LSTM memory span:", memory)
# configuration to remember previous recurrent layer
rs = True

# features is the max length in the corpus, after padding!!!!
# print(X_train.shape)
features = X_txt.shape[2]
print("Generator LSTM learning features:", features)

# batch size
bs = 32
print("Discriminator & Generator learning batch size:", bs)

# number of filters or processing units in CNN
# the number is because of good practices from computer vision
# min 8 max 64, normaly 32 (related to the size of the images)
# 16 for now in this test

# imgage filters
# filters = 16
# filters = 32
filters = 64
# filters = 128
print("Generator CNN filter number:", filters)

disin_shape = X_img[0].shape
genout_shape = X_img[0].shape
# in_shape = (None, None, 1)
# in_shape = (794, 794, 3)
print("Discriminator Input shape:", disin_shape)
print("Generator Output shape:", genout_shape)

ksize = (3,3) 
# ksize = (4,4)
stsize = (1,1)
# stsize = (2,2)
# psize = (5,5)
psize = (2,2)

print("Discriminator & Generator CNN kernel size:", ksize)
print("Discriminator & Generator CNN pad size:", psize)

# neurons/processing units size in the dense layer (THIS SHOULD BE SOM!!!!)
gen_midn =  100*100*3 # 50*50*3 #
gen_reshape = (100,100,3) # (50,50,3) #
print("Generator Dense middle neurons:", gen_midn)
# dn2 = len(XB_set[0])*SECURITY_FACTOR

# numero de neuronas de salida
# out_shape = X_train[0].shape
# out_shape = (None, None, 3)
# out_shape = in_shape
out_dis = y[0].shape[0]
# out_dis = y[0].shape
print("Discriminator Output prediction labels:", out_dis)

channels = img_og_shape[2]
# channels = 8
# dis_midn = filters*out_dis*channels*15*5
dis_midn = filters*channels*out_dis*4
print("Discriminator Dense middle neurons:", dis_midn)

# axtivation functions
in_dis_actf = LeakyReLU(alpha=0.2) # "relu"
in_gen_actf = LeakyReLU(alpha=0.2) # "relu"
hid_ly_actf = LeakyReLU(alpha=0.2) # "relu",
out_dis_act = "sigmoid" # "softmax"
out_gen_act = "tanh" # "softmax"

# loss percentage
dis_ldrop = 0.2
gen_ldrop = 0.2

# padding policy
pad = "same"

# random seed
randseed = 42

# parameters to compile model
# loss function
# ls = "mean_squared_error"
# ls = "categorical_crossentropy"
ls = "binary_crossentropy"

##########################################
# discriminator optimization function
# Adam option
dis_opti = Adam(learning_rate=0.00020, beta_1=0.5)

# Adadelta option
# dis_opti = Adadelta(learning_rate=0.00020)

# Adagrad option
# dis_opti = Adagrad(learning_rate=0.00020, momentum=0.5)

##########################################
# gan/genenerator optimization function
# Adam option
# gan_opti = Adam(learning_rate=0.00020, beta_1=0.5)

# Adadelta option
# gan_opti = Adadelta(learning_rate=0.00030)

# Adagrad option
# gan_opti = Adagrad(learning_rate=0.00020, momentum=0.5)

# SGD option
gan_opti = SGD(learning_rate=0.00020, momentum=0.5)

# evaluation score
# met = ["categorical_accuracy"]
met = ["accuracy"]

# parameters to exeute training
# verbose mode
ver = 0
# training epocha
epo = 500
print("training epochs:", epo)


# %%
# generator layers
# Masking -> LSTM - LSTM -> LSTM -> LSTM -> BatchNorm -> Drop -> Flatten -> Dense -> Reshape -> BatchNorm -> Drop -> Conv2D -> onv2D -> onv2D -> BatchNorm - Conv2D

slim_gen_layers =(

    # input layer (padding and prep)
    Masking(mask_value=0.0, input_shape=(memory, features), name = "GenMaskIn"),

    # intermediate recurrent layer
    LSTM(lstm_units, activation=in_gen_actf, input_shape=(memory, features), return_sequences=rs, name="LSTM_1"),
    # Dropout and Normalization layers
    BatchNormalization(name="GenNorm_1"),
    Dropout(gen_ldrop, name="GenDrop_1"),

    # intermediate recurrent layer
    LSTM(int(lstm_units/2), activation=hid_ly_actf, input_shape=(timesteps, features), return_sequences=rs, name="LSTM_2"),

    # intermediate recurrent layer
    LSTM(int(lstm_units/4), activation=hid_ly_actf, input_shape=(timesteps, features), return_sequences=rs, name="LSTM_3"),

    # intermediate recurrent layer
    # LSTM(int(lstm_units/8), activation=hid_ly_actf, input_shape=(timesteps, features), return_sequences=rs, name="LSTM_4"),

    # Dropout and Normalization layers
    BatchNormalization(name="GenNorm_2"),
    Dropout(gen_ldrop, name="GenDrop_2"),

    # #from 2D to 1D
    Flatten(name="GenLayFlat_1"),
    # mid dense encoding layer
    Dense(gen_midn, activation=hid_ly_actf, name="GenMidDense"),
    # # from 1D to 2D
    Reshape(gen_reshape, name="layReshape_1"),

    # Dropout and Normalization layers
    BatchNormalization(name="GenNorm_3"),
    Dropout(gen_ldrop, name="GenDrop_3"),

    # intermediate convolutional decoder layer
    Conv2D(int(filters/4), ksize, strides=stsize, activation=hid_ly_actf, padding=pad, name="GenConv_1"),
    UpSampling2D(psize, name="GenUpsam_1"),

    # intermediate convolutional decoder layer
    Conv2D(int(filters/2), ksize, strides=stsize, activation=hid_ly_actf, padding=pad, name="GenConv_2"),
    UpSampling2D(psize, name="GenUpsam_2"),

    # Dropout and Normalization layers
    BatchNormalization(name="GenNorm_4"),
    Dropout(gen_ldrop, name="GenDrop_4"),

    # intermediate convolutional decoder layer
    Conv2D(int(filters), ksize, strides=stsize, activation=hid_ly_actf, padding=pad, name="GenConv_3"),
    # UpSampling2D(psize, name="GenUpsam_3"),

    # outputlayer
    Conv2D(channels, ksize, strides=stsize, activation=out_gen_act, input_shape=genout_shape, padding=pad, name="GenOut"),
)


# %%
# defining model
gen_model = Sequential(slim_gen_layers)
gen_model.model_name = gen_model_name


# %%
# NOT compile model
# gen_model.compile(loss=ls, optimizer=gan_opti, metrics=met)
gen_model.summary()


# %%
# discriminator layers
# Input -> Conv2 -> Conv2 -> Conv2 -> BatchNorm -> Drop -> Flatten -> Dense -> dense -> Dense -> Dense -> BatchNorm -> Drop -> Dense
slim_dis_layers = (
    # input layer (padding and prep)
    Input(shape=disin_shape, name="DisLayIn"),

    # intermediate convolutional encoder layer
    Conv2D(filters, ksize, strides=stsize, activation=hid_ly_actf, padding=pad, name="DisConv_1"),
    MaxPooling2D(psize, padding=pad, name="DisPool_1"),

    # Dropout and Normalization layers
    BatchNormalization(name="DisNorm_1"),
    Dropout(gen_ldrop, name="DisDrop_1"),

    # intermediate convolutional encoder layer
    Conv2D(int(filters/2), ksize, strides=stsize, activation=hid_ly_actf, padding=pad, name="DisConv_2"),
    MaxPooling2D(psize, padding=pad, name="DisPool_2"),

    # intermediate convolutional encoder layer
    Conv2D(int(filters/4), ksize, strides=stsize, activation=hid_ly_actf, padding=pad, name="DisConv_3"),
    MaxPooling2D(psize, padding=pad, name="DisPool_3"),

    # Dropout and Normalization layers
    BatchNormalization(name="DisNorm_2"),
    Dropout(gen_ldrop, name="DisDrop_2"),

    # #from 2D to 1D
    Flatten(name="DisLayFlat_1"),

    # mid dense encoding layer
    Dense(dis_midn, activation=hid_ly_actf, name="DisMidDense"),

    # Dropout and Normalization layers
    BatchNormalization(name="DisNorm_3"),
    Dropout(gen_ldrop, name="DisDrop_3"),

    # intermediate dense classification layer
    Dense(int(dis_midn/2), activation=hid_ly_actf, name="DisDense_1"),

    # intermediate dense classification layer
    Dense(int(dis_midn/4), activation=hid_ly_actf, name="DisDense_2"),

    # Dropout and Normalization layers
    BatchNormalization(name="DisNorm_4"),
    Dropout(gen_ldrop, name="DisDrop_4"),

    # intermediate dense classification layer
    Dense(int(dis_midn/8), activation=hid_ly_actf, name="DisDense_3"),

    # output layer, dense time sequential layer.
    Dense(out_dis, activation=out_dis_act, name="DisOut"),
)


# %%
dis_model = Sequential(slim_dis_layers)
dis_model.model_name = dis_model_name


# %%
# compile model
dis_model.compile(loss=ls, optimizer=dis_opti, metrics=met)
dis_model.trainable = False
dis_model.summary()


# %%
# GAN layers
gan_layers = (
    gen_model, 
    dis_model,
)


# %%
gan_model = Sequential(gan_layers)


# %%
gan_model.compile(loss=ls, optimizer=gan_opti, metrics=met)
gan_model.summary()


# %%
# saving model topology into png files
export_model(gen_model, model_fn_path, gen_model_name, timestamp)
export_model(dis_model, model_fn_path, dis_model_name, timestamp)
export_model(gan_model, model_fn_path, gan_model_name, timestamp)


# %%
# config for training
fn_config = (model_fn_path, report_fn_path, dis_model_name, gen_model_name, gan_model_name)
check_epochs = 10


# %%
# dividing according to train/test proportions
# Xt_train, Xt_test, Xi_train, Xi_test = train_test_split(X_txt, X_img, train_size = train_split, test_size = test_split, random_state = randseed)
# Xi_train, Xi_test, y_train, y_test = train_test_split(X_img, y, train_size = train_split, test_size = test_split, random_state = randseed)


# %%
print(X_img.shape, X_txt.shape, y.shape)


# %%
train(gen_model, dis_model, gan_model, X_img, X_txt, y, epo, bs, check_epochs, fn_config)
# train_good(gen_model, dis_model, gan_model, X_img, X_txt, y, epo, bs, check_epochs, fn_config)


# %%



