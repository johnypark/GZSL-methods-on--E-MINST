# -*- coding: utf-8 -*-
# adaopted from this repo
#https://github.com/matusstas/cGAN/tree/main
# and modified to fit the purpose



gpu_info = !nvidia-smi
gpu_info = '\n'.join(gpu_info)
if gpu_info.find('failed') >= 0:
  print('Not connected to a GPU')
else:
  print(gpu_info)


"""# 2. Import Dependencies"""

#!pip install -q wandb

# For general usage
import numpy as np
import tensorflow as tf
import os
import csv

# For dataset
import tensorflow_datasets as tfds

# For models
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers.legacy import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Reshape, LeakyReLU, Dropout, Embedding, Input, Concatenate, Conv2DTranspose

# For callback
from tensorflow.keras.preprocessing.image import array_to_img
from tensorflow.keras.callbacks import Callback

# For streamlining ML workflow
# # import wandb
# from wandb.keras import WandbCallback

# For model memory usage
from tensorflow.keras import backend as K
import humanize

# For visualisation
from matplotlib import pyplot as plt

from IPython import display

# import tensorflow as tf
# import tensorflow_datasets as tfds

# # Load the EMNIST dataset
# dataset = tfds.load('emnist')

# # Print the number of examples in the dataset
# print(f'Number of examples in the dataset: {len(dataset)}')
# train_dataset, val_dataset = dataset.random_split([0.8, 0.2])

from google.colab import drive
drive.mount('/content/drive')

parent_dir = "drive/MyDrive/cGAN"
img_dir = "drive/MyDrive/cGAN/imgs_01"
img_dir_exp = "drive/MyDrive/cGAN/imgs_01/EP2k"
model_dir = "drive/MyDrive/cGAN/model_01"

for path_name in [parent_dir, img_dir, img_dir_exp, model_dir]:
    try:
        os.mkdir(path_name)
    except:
        print('pass')

!ls drive/MyDrive/cGAN

ls drive/MyDrive/cGAN/imgs

"""# 3. Load the dataset"""

# Name of the used dataset
DATASET_NAME = "mnist"

# Shape of the input image
INPUT_SHAPE = (28,28,1)

# Dimension of latent vector
LATENT_VECTOR_DIM = 100

# Number of classes
N_CLASSES = 10

# Number of images in dataset
N_IMAGES = 60000

# Batch size
BATCH_SIZE = 256

# Number of batches that can be formed from a given dataset size
N_BATCHES = N_IMAGES // BATCH_SIZE

# Number of epochs
N_EPOCHS = 2000

# # Name of the project for Weights and Biases platform
# WB_PROJECT = "GAN"

# # Entity (login) for Weights and Biases platform
# WB_ENTITY = "matusstas"

"""### Functions

"""



def preprocess_images(data):
    """ Normalize images to 0-1"""
    image = data['image']
    label = data['label']
    return image / 255, label

def load_dataset():
    """ Load and prepare the dataset """
    # Load the dataset
    dataset = tfds.load(DATASET_NAME, split='train')

    # Take only selected amount of images
    dataset = dataset.take(N_BATCHES * BATCH_SIZE)

    # Preproces images
    dataset = dataset.map(preprocess_images)

    # Cache the dataset for that batch
    dataset = dataset.cache()

    # Shuffle it up
    dataset = dataset.shuffle(N_BATCHES * BATCH_SIZE)

    # Create batches
    dataset = dataset.batch(BATCH_SIZE)

    # Reduces the likelihood of bottlenecking
    dataset = dataset.prefetch(BATCH_SIZE // 2)
    return dataset

dataset = load_dataset()

print(f"shape of the images: {dataset.as_numpy_iterator().next()[0].shape}")
print(f"shape of the labels: {dataset.as_numpy_iterator().next()[1].shape}")

def build_generator():
    """ Build model of the generator using functional API """
    input_label = Input(shape=(1,))
    l = Embedding(N_CLASSES, 50)(input_label)
    l = Dense(7*7)(l)
    l = Reshape((7,7,1))(l)

    input_latent_vector = Input(shape=(LATENT_VECTOR_DIM,))
    lv = Dense(7*7*128)(input_latent_vector)
    lv = LeakyReLU(alpha=0.2)(lv)
    lv = Reshape((7, 7, 128))(lv)

    x = Concatenate()([lv, l])

    x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    output = Conv2D(1, (8,8), activation='sigmoid', padding='same')(x)
    model = Model([input_latent_vector, input_label], output)
    return model

generator = build_generator()



def generate_and_save_images(generator,
                             path_dir,
                                epochs,
                             latent_vectors = seed,
                             **kwargs
                             ):

    N_SAMPLES = 10
    # seed = tf.random.normal([num_examples_to_generate, noise_dim])
    # # Generate labels
    labels = list(range(N_SAMPLES))*N_SAMPLES #np.random.randint(N_CLASSES, size=N_SAMPLES*N_SAMPLES)
    labels = np.expand_dims(labels, axis=-1)

    # Generate images
    images_generated = generator.predict([latent_vectors, labels])
    images_generated = images_generated.reshape(N_SAMPLES, N_SAMPLES, INPUT_SHAPE[0], INPUT_SHAPE[1])

    # Prepare subplots
    fig, ax = plt.subplots(nrows=N_SAMPLES, ncols=N_SAMPLES, figsize=(FIG_SIZE, FIG_SIZE))

    # Change face color of the plot to black
    fig.patch.set_facecolor('xkcd:black')
    for i in range(N_SAMPLES):
      for j in range(N_SAMPLES):
        img = np.squeeze(images_generated[i][j])
        ax[i][j].imshow(img, cmap='gray')
        ax[i][j].axis('off')

    plt.savefig(os.path.join(path_dir,'image_at_epoch_{:04d}.png'.format(epochs)),
                 pil_kwargs = { "quality":75, 'optimize': True})
    plt.close()
    # plt.show()

generator.load_weights()

"""# 4. Build models

## Helper function to calculate model memory usage
"""

def get_model_usage(model):
    """
    Get memory usage of the model with chosen batch size

    modified function from this link: https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    """
    count_shapes_mem = 0
    count_internal_model_mem = 0

    for layer in model.layers:
        layer_type = layer.__class__.__name__

        if layer_type == "Model":
            count_internal_model_mem += get_model_usage(BATCH_SIZE, layer)

        single_layer_mem = 1
        out_shape = layer.output_shape
        out_shape = out_shape[0] if type(out_shape) is list else out_shape

        for shape in out_shape:
            if shape is None:
                continue
            single_layer_mem *= shape
        count_shapes_mem += single_layer_mem

    count_trainable = np.sum([K.count_params(p) for p in model.trainable_weights])
    count_non_trainable = np.sum([K.count_params(p) for p in model.non_trainable_weights])

    # Choose correct precision of floating-point numbers
    precisions = {"float16": 2.0, "float32": 4.0, "float64": 8.0}
    number_size = precisions[K.floatx()]

    total_memory = number_size * (BATCH_SIZE * count_shapes_mem + count_trainable + count_non_trainable)
    total_memory = humanize.naturalsize(total_memory + count_internal_model_mem, binary=True)
    print(f"Total memory usage with batch size of {BATCH_SIZE} is: {total_memory}")

"""## Build generator"""

def build_generator():
    """ Build model of the generator using functional API """
    input_label = Input(shape=(1,))
    l = Embedding(N_CLASSES, 50)(input_label)
    l = Dense(7*7)(l)
    l = Reshape((7,7,1))(l)

    input_latent_vector = Input(shape=(LATENT_VECTOR_DIM,))
    lv = Dense(7*7*128)(input_latent_vector)
    lv = LeakyReLU(alpha=0.2)(lv)
    lv = Reshape((7, 7, 128))(lv)

    x = Concatenate()([lv, l])

    x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    output = Conv2D(1, (8,8), activation='sigmoid', padding='same')(x)
    model = Model([input_latent_vector, input_label], output)
    return model

generator = build_generator()

get_model_usage(generator)

generator.summary()

tf.keras.utils.plot_model(generator,
                          show_shapes=True,
                          to_file = os.path.join(model_dir,
                                                 "cgan_gen.png")
)

"""## Build discriminator"""

def build_discriminator():
    """ Build model of the generator using functional API """
    input_label = Input(shape=(1,))
    l = Embedding(N_CLASSES, 50)(input_label) # change embedding so it matches generator input
    l = Dense(INPUT_SHAPE[0] * INPUT_SHAPE[1])(l)
    l = Reshape((INPUT_SHAPE[0], INPUT_SHAPE[1], 1))(l)

    input_image = Input(shape=INPUT_SHAPE)
    x = Concatenate()([input_image, l])

    x = Conv2D(128, (3,3), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Conv2D(128, (3,3), strides=(2,2), padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Flatten()(x)
    x = Dropout(0.4)(x)
    x = Dense(100)(x)

    output = Dense(1, activation='sigmoid')(x)

    model = Model([input_image, input_label], output)
    return model

discriminator = build_discriminator()

get_model_usage(discriminator)

discriminator.summary()

tf.keras.utils.plot_model(discriminator,
                          show_shapes=True,
                          to_file = os.path.join(model_dir,
                                                 "cgan_discrim.png")
)

"""# Build GAN"""









class GAN(Model):
    def __init__(self, generator, discriminator, *args, **kwargs):
        # Pass through args and kwargs to base class
        super().__init__(*args, **kwargs)

        # Create attributes for gen and disc
        self.generator = generator
        self.discriminator = discriminator

    def compile(self, opt_g, opt_d, loss_g, loss_d, *args, **kwargs):
        # Compile with base class
        super().compile(*args, **kwargs)

        # Create attributes for losses and optimizers
        self.opt_g = opt_g
        self.opt_d = opt_d
        self.loss_g = loss_g
        self.loss_d = loss_d

    @tf.function
    def train_step(self, batch):
        # Get the data
        images_real, labels = batch
        labels = tf.expand_dims(labels, axis=-1)

        # Generate images
        latent_vectors = tf.random.normal(shape=(BATCH_SIZE, LATENT_VECTOR_DIM))
        images_generated = self.generator([latent_vectors, labels], training=False)

        # Train the discriminator
        with tf.GradientTape() as d_tape:
            # Pass the real and fake images to the discriminator model
            yhat_real = self.discriminator([images_real, labels], training=True)
            yhat_fake = self.discriminator([images_generated, labels], training=True)
            yhat_realfake = tf.concat([yhat_real, yhat_fake], axis=0)

            # Create labels for real and fakes images
            y_realfake = tf.concat([tf.zeros_like(yhat_real), tf.ones_like(yhat_fake)], axis=0)

            # Add some noise to the TRUE outputs (crucial step)
            noise_real = 0.15*tf.random.uniform(tf.shape(yhat_real))
            noise_fake = -0.15*tf.random.uniform(tf.shape(yhat_fake))
            y_realfake += tf.concat([noise_real, noise_fake], axis=0)

            # Calculate loss
            total_loss_d = self.loss_d(y_realfake, yhat_realfake)

        # Apply backpropagation
        dgrad = d_tape.gradient(total_loss_d, self.discriminator.trainable_variables)
        self.opt_d.apply_gradients(zip(dgrad, self.discriminator.trainable_variables))

        # Train the generator
        with tf.GradientTape() as g_tape:
            # Generate images
            latent_vectors = tf.random.normal(shape=(BATCH_SIZE, LATENT_VECTOR_DIM))
            images_generated = self.generator([latent_vectors, labels], training=True)

            # Create the predicted labels
            predicted_labels = self.discriminator([images_generated, labels], training=False)

            # Calculate loss - trick to training to fake out the discriminator
            total_loss_g = self.loss_g(tf.zeros_like(predicted_labels), predicted_labels)

        # Apply backpropagation
        ggrad = g_tape.gradient(total_loss_g, self.generator.trainable_variables)
        self.opt_g.apply_gradients(zip(ggrad, self.generator.trainable_variables))

        # Generate after the final epoch
        return {"loss_d":total_loss_d, "loss_g":total_loss_g}

"""# Build Monitoring callback


"""

# better to write a callback
import os

N_SAMPLES = 10
FIG_SIZE = 10
LATENT_VECTOR_DIM = 100
seed = tf.random.normal(shape=(N_SAMPLES*N_SAMPLES, LATENT_VECTOR_DIM))

def generate_and_save_images(generator,
                             path_dir,
                                epochs,
                             latent_vectors = seed,
                             **kwargs
                             ):

    N_SAMPLES = 10
    # seed = tf.random.normal([num_examples_to_generate, noise_dim])
    # # Generate labels
    labels = list(range(N_SAMPLES))*N_SAMPLES #np.random.randint(N_CLASSES, size=N_SAMPLES*N_SAMPLES)
    labels = np.expand_dims(labels, axis=-1)

    # Generate images
    images_generated = generator.predict([latent_vectors, labels])
    images_generated = images_generated.reshape(N_SAMPLES, N_SAMPLES, INPUT_SHAPE[0], INPUT_SHAPE[1])

    # Prepare subplots
    fig, ax = plt.subplots(nrows=N_SAMPLES, ncols=N_SAMPLES, figsize=(FIG_SIZE, FIG_SIZE))

    # Change face color of the plot to black
    fig.patch.set_facecolor('xkcd:black')
    for i in range(N_SAMPLES):
      for j in range(N_SAMPLES):
        img = np.squeeze(images_generated[i][j])
        ax[i][j].imshow(img, cmap='gray')
        ax[i][j].axis('off')

    plt.savefig(os.path.join(path_dir,'image_at_epoch_{:04d}.png'.format(epochs)),
                 pil_kwargs = { "quality":75, 'optimize': True})
    plt.close()
    # plt.show()

class SaveGeneratorOutputs(tf.keras.callbacks.Callback):
    #https://stackoverflow.com/questions/60727279/save-history-of-model-fit-for-different-epochs
    #modified from the above code
    def __init__(self,
                 out_path="./",
                 **kargs):
        super(SaveGeneratorOutputs,self).__init__(**kargs)

        self.out_path = out_path

    def on_epoch_end(self, epoch, logs=None):

        display.clear_output(wait=True)
        generate_and_save_images(self.model.generator, self.out_path, epochs = epoch )


class RecordTraining(tf.keras.callbacks.Callback):
    #https://stackoverflow.com/questions/60727279/save-history-of-model-fit-for-different-epochs
    #modified from the above code
    def __init__(self,
                 out_path="./",
                 csv_name = 'history.csv',
                 model_save = True,
                 save_freq = 100,
                 **kargs):
        super(RecordTraining,self).__init__(**kargs)

        self.out_path = out_path
        self.csv_name = csv_name
        self.model_save = model_save
        self.save_freq = save_freq
        self.full_path_csv = os.path.join( self.out_path, self.csv_name)
        self.full_path_gen = os.path.join( self.out_path, "cgan_gnrtr.h5")
        self.full_path_discrim = os.path.join( self.out_path, "gcan_dscrmntr.h5")

    def on_train_begin(self, logs = None):

        with open(self.full_path_csv,'a') as f:
            y = csv.DictWriter(f,list(logs.keys()))
            y.writeheader()

    def on_epoch_end(self, epoch, logs=None):

        with open(self.full_path_csv,'a') as f:
            y = csv.DictWriter(f,list(logs.keys()))

            logs_mod = dict()

            for key, value in logs.items():
                logs_mod[key] = float(np.mean(value))

            y.writerow(logs_mod)

        if self.model_save == True:
            if (epoch + 1) % self.save_freq == 0:
                self.model.generator.save(self.full_path_gen)
                self.model.discriminator.save(self.full_path_discrim)


## Class reweighting strategies for class imbalance

# class ModelMonitor(Callback):
#     def __init__(self, n_images=10, latent_dim=LATENT_VECTOR_DIM):
#         # Create attributes
#         self.n_images = n_images
#         self.latent_dim = LATENT_VECTOR_DIM

#     def on_epoch_end(self, epoch, logs=None):
#       """
#       After every 10 iterations generate all digits from 0 to 9 and log them
#       into Weights % Biases platform
#       """
#       if epoch % 10 == 0:
#         labels = np.arange(0, N_CLASSES)
#         labels = np.expand_dims(labels, axis=-1)

#         latent_vectors = tf.random.normal(shape=(10, LATENT_VECTOR_DIM))
#         images_generated = generator([latent_vectors, labels])
#         images_generated *= 255

#         for i in range(self.n_images):
#             img = array_to_img(images_generated[i])
#             wandb.log({f"number {i}": wandb.Image(img)})

"""# 5. Train model"""

# wandb.login()

#run = wandb.init(WB_PROJECT="GAN", WB_ENTITY="matusstas")

# Initialize optimizers# Initialize losses# Initialize models
opt_g = Adam(learning_rate=0.0001, beta_1=0.5)
opt_d = Adam(learning_rate=0.00001, beta_1=0.5)

# Initialize losses
loss_g = BinaryCrossentropy()
loss_d = BinaryCrossentropy()

# Initialize models
generator = build_generator()
discriminator = build_discriminator()
gan = GAN(generator, discriminator)
gan.compile(opt_g, opt_d, loss_g, loss_d)

callbacks=[SaveGeneratorOutputs(out_path = img_dir_exp),
           RecordTraining(out_path = model_dir,
                          csv_name = 'cgan_history.csv')
    #ModelMonitor()#,
    # WandbCallback()
]

img_dir_exp

import timeit
start = timeit.timeit()
print("hello")
end = timeit.timeit()
print((end - start)/3600)

import timeit
start = timeit.timeit()

history = gan.fit(dataset, epochs=2000, callbacks=callbacks)
gan.generator.save(os.path.join(model_dir,"cgan_gen.h5"))
gan.discriminator.save(os.path.join(model_dir, "cgan_discrim.h5"))

end = timeit.timeit()
print((end - start)/3600)





"""# 6. Evaluate model"""

# N_SAMPLES = 10
# FIG_SIZE = 10

# # Generate labels
# labels = list(range(10))*10 #np.random.randint(N_CLASSES, size=N_SAMPLES*N_SAMPLES)
# labels = np.expand_dims(labels, axis=-1)

# # Generate images
# latent_vectors = tf.random.normal(shape=(N_SAMPLES*N_SAMPLES, LATENT_VECTOR_DIM))
# images_generated = gan.generator.predict([latent_vectors, labels])
# images_generated = images_generated.reshape(N_SAMPLES, N_SAMPLES, INPUT_SHAPE[0], INPUT_SHAPE[1])

# # Prepare subplots
# fig, ax = plt.subplots(nrows=N_SAMPLES, ncols=N_SAMPLES, figsize=(FIG_SIZE, FIG_SIZE))

# # Change face color of the plot to black
# fig.patch.set_facecolor('xkcd:black')
# for i in range(N_SAMPLES):
#   for j in range(N_SAMPLES):
#     img = np.squeeze(images_generated[i][j])
#     ax[i][j].imshow(img, cmap='gray')
#     ax[i][j].axis('off')

# fig.savefig('output.png', dpi=300)





"""# 7. Save model's weights"""

gan.save_weights('./checkpoints/my_checkpoint')

"""# 8. Load model's weights

In order to load weights, model has to be compiled
"""

# Initialize optimizers
opt_g = Adam(learning_rate=0.0001, beta_1=0.5)
opt_d = Adam(learning_rate=0.00001, beta_1=0.5)

# Initialize losses
loss_g = BinaryCrossentropy()
loss_d = BinaryCrossentropy()

# Initialize models
generator = build_generator()
discriminator = build_discriminator()
gan = GAN(generator, discriminator)
gan.compile(opt_g, opt_d, loss_g, loss_d)

gan.load_weights('./checkpoints/my_checkpoint')

"""# Save model in HDF5 format

Only generator is essential for us
"""

gan.generator.save("cgan.h5")

"""# Load model in HDF5 format"""

generator = load_model('cgan.h5')