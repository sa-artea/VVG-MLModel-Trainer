"""
https://machinelearningmastery.com/how-to-develop-a-generative-adversarial-network-for-an-mnist-handwritten-digits-from-scratch-in-keras/
"""

# example of training a gan on mnist
from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import vstack
from numpy.random import randn
from numpy.random import randint
from keras.datasets.mnist import load_data
from keras.optimizers import Adam
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Reshape
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv2DTranspose
from keras.layers import LeakyReLU
from keras.layers import Dropout
from matplotlib import pyplot

# define the standalone discriminator model

def define_discriminator(in_shape=(28, 28, 1)):
	model = Sequential()
	model.add(Conv2D(64, (3, 3), strides=(2, 2),
	          padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Conv2D(64, (3, 3), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Dropout(0.4))
	model.add(Flatten())
	model.add(Dense(1, activation='sigmoid'))
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model

# define the standalone generator model


def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	# upsample to 14x14
	model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	# upsample to 28x28
	model.add(Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7, 7), activation='sigmoid', padding='same'))
	return model

# define the combined generator and discriminator model, for updating the generator


def define_gan(g_model, d_model):
	# make weights in the discriminator not trainable
	d_model.trainable = False
	# connect them
	model = Sequential()
	# add generator
	model.add(g_model)
	# add the discriminator
	model.add(d_model)
	# compile model
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model

# load and prepare mnist training images


def load_real_samples():
	# load mnist dataset
	(trainX, _), (_, _) = load_data()
	# expand to 3d, e.g. add channels dimension
	X = expand_dims(trainX, axis=-1)
	# convert from unsigned ints to floats
	X = X.astype('float32')
	# scale from [0,255] to [0,1]
	X = X / 255.0
	return X

# select real samples


def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = randint(0, dataset.shape[0], n_samples)
	# retrieve selected images
	X = dataset[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, 1))
	return X, y

# generate points in latent space as input for the generator


def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input

# use the generator to generate n fake examples, with class labels


def generate_fake_samples(g_model, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = g_model.predict(x_input)
	# create 'fake' class labels (0)
	y = zeros((n_samples, 1))
	return X, y

# create and save a plot of generated images (reversed grayscale)


def save_plot(examples, epoch, n=10):
	# plot images
	for i in range(n * n):
		# define subplot
		pyplot.subplot(n, n, 1 + i)
		# turn off axis
		pyplot.axis('off')
		# plot raw pixel data
		pyplot.imshow(examples[i, :, :, 0], cmap='gray_r')
	# save plot to file
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	pyplot.savefig(filename)
	pyplot.close()

# evaluate the discriminator, plot generated images, save generator model


def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
	# prepare real samples
	X_real, y_real = generate_real_samples(dataset, n_samples)
	# evaluate discriminator on real examples
	_, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
	# prepare fake examples
	x_fake, y_fake = generate_fake_samples(g_model, latent_dim, n_samples)
	# evaluate discriminator on fake examples
	_, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
	# summarize discriminator performance
	print('>Accuracy real: %.0f%%, fake: %.0f%%' % (acc_real*100, acc_fake*100))
	# save plot
	save_plot(x_fake, epoch)
	# save the generator model tile file
	filename = 'generator_model_%03d.h5' % (epoch + 1)
	g_model.save(filename)

# train the generator and discriminator


def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=256):
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)
	# manually enumerate epochs
	for i in range(n_epochs):
		# enumerate batches over the training set
		for j in range(bat_per_epo):
			# get randomly selected 'real' samples
			X_real, y_real = generate_real_samples(dataset, half_batch)
			# generate 'fake' examples
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			# create training set for the discriminator
			X, y = vstack((X_real, X_fake)), vstack((y_real, y_fake))
			# update discriminator model weights
			d_loss, _ = d_model.train_on_batch(X, y)
			# prepare points in latent space as input for the generator
			X_gan = generate_latent_points(latent_dim, n_batch)
			# create inverted labels for the fake samples
			y_gan = ones((n_batch, 1))
			# update the generator via the discriminator's error
			g_loss = gan_model.train_on_batch(X_gan, y_gan)
			# summarize loss on this batch
			print('>%d, %d/%d, d=%.3f, g=%.3f' %
			      (i+1, j+1, bat_per_epo, d_loss, g_loss))
		# evaluate the model performance, sometimes
		if (i+1) % 10 == 0:
			summarize_performance(i, g_model, d_model, dataset, latent_dim)


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
    cond14 = LSTM(int(lbl_lstm/2,
                      activation=lbl_ly_actf,
                      input_shape=mem_shape,
                      return_sequences=lbl_rs,
                      name="TxtCDisLblLSTM_14")(cond6t)

                  # batch normalization + drop layers to avoid overfit for img
                  cond15=BatchNormalization(name="TxtCDisLblBN_14")(cond14)
                  cond16=Dropout(hid_ldrop, name="TxtCDisLblDrop_15")(cond15)

                  # intermediate LSTM layer for text
                  cond17=LSTM(lbl_lstm,
                              activation=lbl_ly_actf,
                              input_shape=mem_shape,
                              return_sequences=lbl_rs,
                              name="TxtCDisLblLSTM_19")(cond16)

                  # LAYER CREATION
                  # input layer
                  in_img=Input(shape=img_shape, name="CDisImgIn")

                  concat_img=Concatenate(
                      axis=-1, name="ImgCDisConcat_21")([in_img, cond13])

                  # DISCRIMINATOR LAYERS
                  # intermediate conv layer
                  lyr1=Conv2D(in_filters, kernel_size=in_ksize,
                              padding=in_pad, activation=in_lyr_act,
                              strides=in_stsize, name="ImgCDisConv2D_22")(concat_img)

                  # intermediate conv layer
                  lyr2=Conv2D(int(filters/2), kernel_size=ksize,
                              padding=pad, activation=hid_lyr_act,
                              strides=stsize, name="ImgCDisConv2D_23")(lyr1)

                  # batch normalization + drop layers to avoid overfit
                  lyr3=BatchNormalization(name="ImgCDisBN_24")(lyr2)
                  lyr4=Dropout(hid_ldrop, name="ImgCDisDrop_25")(lyr3)

                  # intermediate conv layer
                  lyr5=Conv2D(int(filters/4), kernel_size=ksize,
                              padding=pad, activation=hid_lyr_act,
                              strides=stsize, name="ImgCDisConv2D_26")(lyr4)

                  # intermediate conv layer
                  lyr6=Conv2D(int(filters/8), kernel_size=ksize,
                              padding=pad, activation=hid_lyr_act,
                              strides=stsize, name="ImgCDisConv2D_27")(lyr5)

                  # batch normalization + drop layers to avoid overfit
                  lyr7=BatchNormalization(name="ImgCDisBN_28")(lyr6)
                  lyr8=Dropout(hid_ldrop, name="ImgCDisDrop_29")(lyr7)

                  # flatten from 2D to 1D
                  lyr9=Flatten(name="ImgCDisFlat_29")(lyr8)

                  #TXT DISCRIMINATOR
                  # LAYER CREATION
                  # input layer
                  in_txt=Input(shape=txt_shape, name="CDisTxtIn")

                  # concat txt input with labels conditional
                  concat_txt=Concatenate(
                      axis=-1, name="TxtCDisConcat_31")([in_txt, cond17])

                  # DISCRIMINATOR LAYERS
                  # masking input text
                  lyr10=Masking(mask_value=mval, input_shape=txt_shape,
                                name="TxtCDisMask_32")(concat_txt)  # concat1

                  # input LSTM layer
                  lyr11=LSTM(in_lstm, activation=in_lyr_act,
                             input_shape=txt_shape,
                             return_sequences=in_rs,
                             name="TxtDisLSTM_33")(lyr10)

                  # batch normalization + drop layers to avoid overfit
                  lyr12=BatchNormalization(name="TxtDisBN_34")(lyr11)
                  lyr13=Dropout(hid_ldrop, name="TxtDisDrop_4")(lyr12)

                  # intermediate LSTM layer
                  lyr14=LSTM(int(lstm_units/2),
                             activation=hid_lyr_act,
                             input_shape=mem_shape,
                             return_sequences=rs,
                             name="TxtDisLSTM_5")(lyr13)

                  # intermediate LSTM layer
                  lyr15=LSTM(int(lstm_units/4),
                             activation=hid_lyr_act,
                             input_shape=mem_shape,
                             return_sequences=rs,
                             name="TxtDisLSTM_6")(lyr14)

                  # batch normalization + drop layers to avoid overfit
                  lyr16=BatchNormalization(name="TxtDisBN_7")(lyr15)
                  lyr17=Dropout(hid_ldrop, name="TxtDisDrop_8")(lyr16)

                  # flatten from 2D to 1D
                  lyr18=Flatten(name="TxtDisFlat_9")(lyr17)

                  # concat img encoding + txt encoding
                  concat_encoding=Concatenate(
                      axis=-1, name="DenseCDisConcat_31")([lyr18, lyr9])

                  # dense classifier layers
                  lyr19=Dense(int(mid_disn), activation=hid_cls_act,
                              name="TxtDisDense_10")(concat_encoding)
                  lyr20=Dense(int(mid_disn/2), activation=hid_cls_act,
                              name="TxtDisDense_11")(lyr19)
                  # drop layer
                  lyr21=Dropout(hid_ldrop, name="TxtDisDrop_12")(lyr20)

                  # dense classifier layers
                  lyr22=Dense(int(mid_disn/4), activation=hid_cls_act,
                              name="TxtDisDense_13")(lyr21)
                  lyr23=Dense(int(mid_disn/8), activation=hid_cls_act,
                              name="TxtDisDense_14")(lyr22)
                  # drop layer
                  lyr24=Dropout(hid_ldrop, name="TxtDisDrop_15")(lyr23)

                  # dense classifier layers
                  lyr25=Dense(int(mid_disn/16), activation=hid_cls_act,
                              name="TxtDisDense_16")(lyr24)
                  lyr26=Dense(int(mid_disn/32), activation=hid_cls_act,
                              name="TxtDisDense_17")(lyr25)

                  # output layer
                  out_cls=Dense(out_nsize, activation=out_lyr_act,
                                name="TxtDisOut")(lyr26)

                  # model definition
                  model=Model(
                      inputs=[in_img, in_txt, in_labels], outputs=out_cls)

                  return model








# size of the latent space
latent_dim = 100
# create the discriminator
d_model = define_discriminator()
# create the generator
g_model = define_generator(latent_dim)
# create the gan
gan_model = define_gan(g_model, d_model)
# load image data
dataset = load_real_samples()
# train model
train(g_model, d_model, gan_model, dataset, latent_dim)
