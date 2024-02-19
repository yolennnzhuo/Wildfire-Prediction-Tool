import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Definition of the Variational Autoencoder (VAE) class.
class VAE(tf.keras.Model):
    """
    Variational Autoencoder (VAE) model for image generation and reconstruction.

    Attributes:
        latent_dim: The dimensionality of the latent space.
        input_dim: The dimensionality of the input data.
        encoder: The encoder model.
        decoder: The decoder model.
        vae: The full VAE model.

    Methods:
        build_encoder(): Build the encoder part of the VAE.
        build_decoder(): Build the decoder part of the VAE.
        build_vae(learning_rate=0.0001, reconstruction_weight=0.5, kl_weight=0.5): Build the full VAE model.
        sampling(args): Reparameterization trick to sample from the latent space.
        loss_function(inputs, outputs, z_mean, z_log_var, reconstruction_weight=0.5, kl_weight=0.5): Compute the loss function for the VAE.
        train(x_train, y_train, batch_size, epochs, x_test, y_test): Train the VAE model on the provided data.
        plot_loss(): Plot the training loss.
    """

    def __init__(self, latent_dim=128, input_dim=256*256*1, learning_rate=0.0001, reconstruction_weight=0.5, kl_weight=0.5):
        super(VAE, self).__init__()
        
        """     
        :param latent_dim: The dimensionality of the latent space.
        :type latent_dim: int, optional
        :param input_dim: The dimensionality of the input data.
        :type input_dim: int, optional
        :param learning_rate: The learning rate for model training.
        :type learning_rate: float, optional
        :param reconstruction_weight: The weight for the reconstruction loss term in the VAE loss function.
        :type reconstruction_weight: float, optional
        :param kl_weight: The weight for the Kullback-Leibler (KL) divergence loss term in the VAE loss function.
        :type kl_weight: float, optional
        """        
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.vae = self.build_vae(learning_rate=learning_rate, reconstruction_weight=reconstruction_weight, kl_weight=kl_weight)

    def build_encoder(self):
        """
        Build the encoder part of the autoencoder.

        :return: The encoder model.
        :rtype: tf.keras.Model
        """
        inputs = tf.keras.Input(shape=(self.input_dim,))
        x = layers.Dense(1024, activation='relu')(inputs)
        x = layers.Dense(512, activation='relu')(x)
        z_mean = layers.Dense(self.latent_dim)(x)
        z_log_var = layers.Dense(self.latent_dim)(x)
        z = layers.Lambda(self.sampling, output_shape=(self.latent_dim,))([z_mean, z_log_var])
        return tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')

    def build_decoder(self):
        """
        Build the decoder part of the autoencoder.

        :return: The decoder model.
        :rtype: tf.keras.Model
        """
        latent_inputs = tf.keras.Input(shape=(self.latent_dim,))
        x = layers.Dense(512, activation='relu')(latent_inputs)
        x = layers.Dense(1024, activation='relu')(x)
        outputs = layers.Dense(self.input_dim, activation='sigmoid')(x)
        return tf.keras.Model(latent_inputs, outputs, name='decoder')

    def build_vae(self, learning_rate=0.0001, reconstruction_weight=0.5, kl_weight=0.5):
        """
        Build the full autoencoder model.

        :param learning_rate: The learning rate for the optimizer.
        :type learning_rate: float, optional
        :param reconstruction_weight: The weight for the reconstruction loss term in the VAE loss function.
        :type reconstruction_weight: float, optional
        :param kl_weight: The weight for the Kullback-Leibler (KL) divergence loss term in the VAE loss function.
        :type kl_weight: float, optional
        :return: The full autoencoder model.
        :rtype: tf.keras.Model
        """
        encoder_inputs = tf.keras.Input(shape=(self.input_dim,))
        z_mean, z_log_var, z = self.encoder(encoder_inputs)
        decoder_outputs = self.decoder(z)
        vae = tf.keras.Model(encoder_inputs, decoder_outputs, name='vae')
        vae.add_loss(self.loss_function(encoder_inputs, decoder_outputs, z_mean, z_log_var,reconstruction_weight=reconstruction_weight, kl_weight=kl_weight))
        vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate))
        return vae
        
    # Reparameterization trick
    def sampling(self, args):
        """
        Reparameterization trick to sample from the latent space.

        :param args: Tuple containing the mean and log variance tensors of the latent space.
        :type args: tuple
        :return: The sampled data from the latent space.
        :rtype: tf.Tensor 
        """
        z_mean, z_log_var = args
        epsilon = tf.keras.backend.random_normal(shape=(tf.keras.backend.shape(z_mean)[0], self.latent_dim), mean=0., stddev=1.)
        return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon

    # Custom loss function
    def loss_function(self, inputs, outputs, z_mean, z_log_var, reconstruction_weight=0.5, kl_weight=0.5):
        """
        Compute the loss function for the Variational Autoencoder (VAE).

        :param inputs: The input tensor representing the original data.
        :type inputs: tf.Tensor
        :param outputs: The output tensor representing the reconstructed data.
        :type outputs: tf.Tensor
        :param z_mean: The tensor representing the mean of the latent space.
        :type z_mean: tf.Tensor
        :param z_log_var: The tensor representing the log variance of the latent space.
        :type z_log_var: tf.Tensor
        :param reconstruction_weight: The weight for the reconstruction loss term in the VAE loss function.
        :type reconstruction_weight: float, optional
        :param kl_weight: The weight for the Kullback-Leibler (KL) divergence loss term in the VAE loss function.
        :type kl_weight: float, optional
        :return: The computed loss value.
        :rtype: tf.Tensor
        """
        reconstruction_loss = tf.keras.losses.binary_crossentropy(tf.keras.backend.flatten(inputs), tf.keras.backend.flatten(outputs))
        reconstruction_loss *= self.input_dim  # Adjust for image size
        kl_loss = 1 + z_log_var - tf.keras.backend.square(z_mean) - tf.keras.backend.exp(z_log_var)
        kl_loss = tf.keras.backend.mean(kl_loss, axis=-1)
        kl_loss *= -0.5
        return tf.keras.backend.mean(reconstruction_weight * reconstruction_loss + kl_weight * kl_loss)

    def train(self, x_train, y_train, batch_size, epochs, x_test, y_test):
        """
        Train the VAE model on the provided training data.

        :param x_train: The input training data.
        :type x_train: np.ndarray or tf.Tensor
        :param y_train: The target training data.
        :type y_train: np.ndarray or tf.Tensor
        :param batch_size: The batch size for training.
        :type batch_size: int
        :param epochs: The number of epochs for training.
        :type epochs: int
        :param x_test: The input test data.
        :type x_test: np.ndarray or tf.Tensor
        :param y_test: The target test data.
        :type y_test: np.ndarray or tf.Tensor
        """
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
        his = self.vae.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data = (x_test, y_test), callbacks=[early_stop], shuffle=True)
        self.his = his

    def plot_loss(self):
        """
        Plot the training loss.
        """
        plt.plot(self.his.history['loss'])
        plt.plot(self.his.history['val_loss'])
        plt.legend(['Loss', 'Val_loss'])
        plt.show()