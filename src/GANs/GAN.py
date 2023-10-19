# https://www.tensorflow.org/tutorials/generative/dcgan
# adopted from https://colab.research.google.com/drive/1JigtAVWVgDCbTX9yrrLZpgGr-NMbyOLo
# modified to fit the purpose of this repo

import tensorflow as tf

tf.__version__

# To generate GIFs
!pip install imageio
!pip install git+https://github.com/tensorflow/docs

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time

from IPython import display

"""### Load and prepare the dataset

You will use the MNIST dataset to train the generator and the discriminator. The generator will generate handwritten digits resembling the MNIST data.
"""

(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256

# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

"""## Create the models

Both the generator and discriminator are defined using the [Keras Sequential API](https://www.tensorflow.org/guide/keras#sequential_model).

### The Generator

The generator uses `tf.keras.layers.Conv2DTranspose` (upsampling) layers to produce an image from a seed (random noise). Start with a `Dense` layer that takes this seed as input, then upsample several times until you reach the desired image size of 28x28x1. Notice the `tf.keras.layers.LeakyReLU` activation for each layer, except the output layer which uses tanh.
"""

def make_generator_model(INPUT = 10):
    model = tf.keras.Sequential()


    model.add(tf.keras.layers.InputLayer(input_shape=(INPUT,)))
    # model.add(layers.Dense(32, use_bias=False, input_shape=(INPUT,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    model.add(layers.Dense(64, use_bias=False, input_shape=(INPUT,)))
    # model.add(layers.Dense(64, use_bias=False))
    model.add(layers.Reshape((1, 1, 64)))
    # model.add(layers.Dense(256, use_bias=False, input_shape=(INPUT,)))
    # model.add(layers.BatchNormalization())
    # model.add(layers.LeakyReLU())

    # assert model.output_shape == (None, 3, 3, 64)  # Note: None is the batch size

    model.add(layers.Conv2DTranspose(64, (7, 7), strides=(2, 2), padding='valid', use_bias=False))
    #assert model.output_shape == (None, 7, 7, 64)
    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    #assert model.output_shape == (None, 7, 7, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())


    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    #assert model.output_shape == (None, 14, 14, 32)
    model.add(layers.Conv2DTranspose(32, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    #assert model.output_shape == (None, 7, 7, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(16, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    #assert model.output_shape == (None, 7, 7, 64)
    model.add(layers.Conv2DTranspose(16, (3, 3), strides=(1, 1), padding='same', use_bias=False))
    #assert model.output_shape == (None, 7, 7, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    #model.add(layers.Conv2DTranspose(1, (3, 3), strides=(2, 2), padding='valid', use_bias=False))
    model.add(layers.Conv2DTranspose(1, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    #assert model.output_shape == (None, 28, 28, 1)

    return model

"""Use the (as yet untrained) generator to create an image."""



"""### The Discriminator

The discriminator is a CNN-based image classifier.
"""

def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.Conv2D(16, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.Conv2D(32, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.Conv2D(64, (3, 3), strides=(1, 1), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.MaxPooling2D(pool_size = (2,2)))
    model.add(layers.Dropout(0.3))

    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(64))
    model.add(layers.Dense(1))

    return model

generator = make_generator_model()

noise = tf.random.normal([1, 10])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
discriminator = make_discriminator_model()

discriminator.summary()

"""Use the (as yet untrained) discriminator to classify the generated images as real or fake. The model will be trained to output positive values for real images, and negative values for fake images."""

discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print (decision)

"""## Define the loss and optimizers

Define loss functions and optimizers for both models.

"""

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

"""### Discriminator loss

This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
"""

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

"""### Generator loss
The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, compare the discriminators decisions on the generated images to an array of 1s.
"""

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

"""The discriminator and the generator optimizers are different since you will train two networks separately."""

generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1 = .5)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-5, beta_1 = .5)

"""### Save checkpoints
This notebook also demonstrates how to save and restore models, which can be helpful in case a long running training task is interrupted.
"""

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

"""## Define the training loop

"""

EPOCHS = 200
noise_dim = 10
num_examples_to_generate = 64

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

"""The training loop begins with generator receiving a random seed as input. That seed is used to produce an image. The discriminator is then used to classify real images (drawn from the training set) and fakes images (produced by the generator). The loss is calculated for each of these models, and the gradients are used to update the generator and discriminator."""

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

    return gen_loss, disc_loss

from tqdm import tqdm

def train(dataset, epochs):
  ls_gen_loss =[]
  ls_disc_loss = []
  for epoch in range(epochs):
    start = time.time()

    for image_batch in tqdm(dataset):

      gen_loss, disc_loss = train_step(image_batch)
      ls_gen_loss += [gen_loss]
      ls_disc_loss += [disc_loss]

    # Produce images for the GIF as you go
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))
    print(f'Generator loss:{gen_loss}\nDiscriminator loss:{disc_loss}')

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)
  return ls_gen_loss, ls_disc_loss

"""**Generate and save images**

"""

def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)

  fig = plt.figure(figsize=(8, 8))

  for i in range(predictions.shape[0]):
      plt.subplot(8, 8, i+1)
      plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')

  plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  plt.show()

"""## Train the model
Call the `train()` method defined above to train the generator and discriminator simultaneously. Note, training GANs can be tricky. It's important that the generator and discriminator do not overpower each other (e.g., that they train at a similar rate).

At the beginning of the training, the generated images look like random noise. As training progresses, the generated digits will look increasingly real. After about 50 epochs, they resemble MNIST digits. This may take about one minute / epoch with the default settings on Colab.
"""

gen_loss, disc_loss = train(train_dataset, EPOCHS)

ls_gen_loss = [i.numpy() for i in gen_loss]
ls_disc_loss = [i.numpy() for i in disc_loss]
plt.plot(range(len(ls_gen_loss)), ls_gen_loss)
plt.plot(range(len(ls_disc_loss)), ls_disc_loss)

import pandas as pd
pd.DataFrame({"generator_loss":ls_gen_loss, "discrim_loss":ls_disc_loss}).to_csv("dcGAN_train_result_Oct152023.csv")

"""plt.plot(range(len(ls_gen_loss)), ls_gen_loss)
Restore the latest checkpoint.
"""

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

"""## Create a GIF

"""

# Display a single image using the epoch number
def display_image(epoch_no):
  return PIL.Image.open('image_at_epoch_{:04d}.png'.format(epoch_no))

display_image(EPOCHS)

"""Use `imageio` to create an animated gif using the images saved during training."""

anim_file = 'dcgan.gif'

with imageio.get_writer(anim_file, mode='I') as writer:
  filenames = glob.glob('image*.png')
  filenames = sorted(filenames)
  for filename in filenames:
    image = imageio.imread(filename)
    writer.append_data(image)
  image = imageio.imread(filename)
  writer.append_data(image)

import datetime
now = datetime.datetime.now()

import tensorflow_docs.vis.embed as embed
embed.embed_file(anim_file)

noise = tf.random.normal([1, 10])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')

generated_image.shape

tf.random.normal([1, 100])

generator.summary()

"""## Next steps

This tutorial has shown the complete code necessary to write and train a GAN. As a next step, you might like to experiment with a different dataset, for example the Large-scale Celeb Faces Attributes (CelebA) dataset [available on Kaggle](https://www.kaggle.com/jessicali9530/celeba-dataset). To learn more about GANs see the [NIPS 2016 Tutorial: Generative Adversarial Networks](https://arxiv.org/abs/1701.00160).
"""