from numpy import expand_dims
from numpy import zeros
from numpy import ones
from numpy import asarray
from numpy.random import randn
from numpy.random import randint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv1D, MaxPooling1D,Convolution1D,MaxPool1D,UpSampling1D,BatchNormalization
from tensorflow.keras import backend
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import numpy as np



# load the images
def load_real_samples():
    # load dataset
    #(trainX, trainy), (_, _) = load_data()
    dfDS = pd.read_csv('./dataset/pcapdroid_10class_each_normalized_cuttedfloefeature.csv')
    X_full = dfDS.iloc[:, 1:len(dfDS.columns)].values
    Y_full = dfDS["label"].values
    # expand to 3d, e.g. add channels
    X = expand_dims(X_full, axis=-1)
    # convert from ints to floats
    #X = X.astype('float32')
    # scale from [0,255] to [-1,1]
    #X = (X - 127.5) / 127.5
    Y_full=Y_full.reshape(Y_full.shape[0],1)
    print("Y_full",Y_full)

    print("load_real_samples()",X.shape, Y_full.shape)
    return [X, Y_full]


# custom activation function
def custom_activation(output):
    logexpsum = backend.sum(backend.exp(output), axis=-1, keepdims=True)
    result = logexpsum / (logexpsum + 1.0)
    return result

# define the standalone supervised and unsupervised discriminator models
def define_discriminator(in_shape=(66,1), n_classes=10):
    # image input
    in_image = Input(shape=in_shape)
    # downsample
    fe = Convolution1D(64,3,padding="same",activation="relu",input_shape=in_shape,name='conv_1')(in_image)
    fe = Convolution1D(64,3,padding="same",activation="relu",name='conv_2')(fe)
    fe = MaxPooling1D(pool_size=(2),name='maxpool_1')(fe)
    # downsample
    fe = Convolution1D(128,3,padding="same",activation="relu",input_shape=in_shape,name='conv_3')(fe)
    fe = Convolution1D(128,3,padding="same",activation="relu",name='conv_4')(fe)
    fe = MaxPool1D(pool_size=(2),name='maxpool_2')(fe)
    #mlp
    fe = Flatten()(fe)
    fe = Dense(128,activation="relu",name='dense_1')(fe)
    fe = Dropout(0.5)(fe)
    fe = Dense(n_classes,name='dense_2')(fe)
    c_out_layer = Activation('softmax')(fe)

    # define and compile supervised discriminator model
    c_model = Model(in_image, c_out_layer)
    c_model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(0.0002, 0.1), metrics=['accuracy'])
    # unsupervised output
    d_out_layer = Lambda(custom_activation)(fe)
    # define and compile unsupervised discriminator model
    d_model = Model(in_image, d_out_layer)
    d_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.1))
    return d_model, c_model

# define the standalone generator model
def define_generator(latent_dim):
    # image generator input
    in_lat = Input(shape=(latent_dim,))
    # foundation for 7x7 image
    n_nodes = 128 * 33
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((33,128))(gen)
    gen = BatchNormalization(momentum=0.8)(gen)
    gen = UpSampling1D()(gen)
    out_layer = Conv1D(1, 3, activation='tanh', padding='same')(gen)
    # define model
    model = Model(in_lat, out_layer)
    return model

# define the combined generator and discriminator model, for updating the generator
def define_gan(g_model, d_model):
    # make weights in the discriminator not trainable
    d_model.trainable = False
    # connect image output from generator as input to discriminator
    gan_output = d_model(g_model.output)
    # define gan model as taking noise and outputting a classification
    model = Model(g_model.input, gan_output)
    # compile model
    opt = Adam(0.0002, 0.1)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model



# select a supervised subset of the dataset, ensures classes are balanced
def select_supervised_samples(dataset, n_samples=10000, n_classes=10):
    X, y = dataset
    y=np.transpose(y)
    X_list, y_list = list(), list()
    n_per_class = int(n_samples / n_classes)

    for i in range(n_classes):
        # get all images for this class
        X_with_class = X[y[0] == i]
        print("------------X_with_class------------",X_with_class.shape)
        # choose random instances
        ix = randint(0, len(X_with_class), n_per_class)
        # add to list
        [X_list.append(X_with_class[j]) for j in ix]
        [y_list.append(i) for j in ix]
    return asarray(X_list), asarray(y_list)

# select real samples
def generate_real_samples(dataset, n_samples):
    # split into images and labels
    images, labels = dataset
    # choose random instances
    ix = randint(0, images.shape[0], n_samples)
    # select images and labels
    X, labels = images[ix], labels[ix]
    # generate class labels
    y = ones((n_samples, 1))
    return [X, labels], y

# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
    # generate points in the latent space
    z_input = randn(latent_dim * n_samples)
    # reshape into a batch of inputs for the network
    z_input = z_input.reshape(n_samples, latent_dim)
    return z_input

# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
    # generate points in latent space
    z_input = generate_latent_points(latent_dim, n_samples)
    # predict outputs
    images = generator.predict(z_input)
    # create class labels
    y = zeros((n_samples, 1))
    return images, y

# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, c_model, latent_dim, dataset, n_samples=100):
    # prepare fake examples
    X, _ = generate_fake_samples(g_model, latent_dim, n_samples)
    # scale from [-1,1] to [0,1]
    X = (X + 1) / 2.0
    # # plot images
    # 保存生成图片
    # for i in range(100):
    #     # define subplot
    #     pyplot.subplot(10, 10, 1 + i)
    #     # turn off axis
    #     pyplot.axis('off')
    #     # plot raw pixel data
    #     pyplot.imshow(X[i, :, :, 0], cmap='gray_r')
    # # save plot to file
    # filename1 = 'generated_plot_%04d.png' % (step+1)
    # pyplot.savefig(filename1)
    # pyplot.close()

    # evaluate the classifier model
    X, y = dataset
    _, acc = c_model.evaluate(X, y, verbose=0)
    print('Classizer Accuracy: %.3f%%' % (acc * 100))
    # save the generator model
    filename2 = './SGAN/cross_g_model_%04d.h5' % (step+1)
    g_model.save(filename2)
    # save the classifier model
    filename3 = './SGAN/cross_c_model_%04d.h5' % (step+1)
    c_model.save(filename3)
    print('>Saved:  %s, and %s' % ( filename2, filename3))




# train the generator and discriminator
def train(g_model, d_model, c_model, gan_model, dataset, latent_dim, n_epochs=20, n_batch=128):
    # select supervised dataset
    X_sup, y_sup = select_supervised_samples(dataset)
    # calculate the number of batches per training epoch
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    # calculate the number of training iterations
    #n_steps = bat_per_epo * n_epochs
    n_steps=5000  #3000 96.69
    # calculate the size of half a batch of samples
    half_batch = int(n_batch / 2)
    print('n_epochs=%d, n_batch=%d, 1/2=%d, b/e=%d, steps=%d' % (n_epochs, n_batch, half_batch, bat_per_epo, n_steps))
    # manually enumerate epochs
    for i in range(n_steps):
        curstepstarttime=datetime.datetime.now()
        print("--------cur step-----------",i,"----------n_steps------------",n_steps)
        # update supervised discriminator (c)
        [Xsup_real, ysup_real], _ = generate_real_samples([X_sup, y_sup], half_batch)
        c_loss, c_acc = c_model.train_on_batch(Xsup_real, ysup_real)
        # update unsupervised discriminator (d)
        [X_real, _], y_real = generate_real_samples(dataset, half_batch)
        #print("dataset",dataset[0].shape,dataset[1].shape)
        d_loss1 = d_model.train_on_batch(X_real, y_real)
        X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
        d_loss2 = d_model.train_on_batch(X_fake, y_fake)
        # update generator (g)
        X_gan, y_gan = generate_latent_points(latent_dim, n_batch), ones((n_batch, 1))
        g_loss = gan_model.train_on_batch(X_gan, y_gan)
        # summarize loss on this batch
        print('>%d, c[%.3f,%.0f], d[%.3f,%.3f], g[%.3f]' % (i+1, c_loss, c_acc*100, d_loss1, d_loss2, g_loss))
        curstepstoptime=datetime.datetime.now()
        print("当前step执行时间(s)：",(curstepstoptime - curstepstarttime).microseconds,'\n')
        dLosses.append(d_loss2)
        gLosses.append(g_loss)
        # evaluate the model performance every so often
        if (i+1) % 100 == 0:
            summarize_performance(i, g_model, c_model, latent_dim, dataset)
    plotLoss(n_steps)

# Plot the loss from each batch
def plotLoss(epoch):
    plt.figure(figsize=(10, 8))
    plt.plot(dLosses, label='Discriminitive loss')
    plt.plot(gLosses, label='Generative loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./SGAN/gan_loss_epoch_%d.png' % epoch)
dLosses = []
gLosses = []



if __name__ == '__main__':
   # step1 加载数据集
   dataset = load_real_samples()
   #step2 构建模型
   latent_dim=100
   d_model, c_model = define_discriminator()
   print("-----------------d_model----------------------:")
   d_model.summary()
   print("-----------------c_model----------------------:")
   c_model.summary()
   # create the generator
   g_model = define_generator(latent_dim)
   print("-----------------g_model----------------------:")
   g_model.summary()
   # create the gan
   gan_model = define_gan(g_model, d_model)


   train(g_model, d_model, c_model, gan_model, dataset, latent_dim)