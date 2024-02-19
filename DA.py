import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy.linalg import inv
import tensorflow as tf
from tensorflow.keras.optimizers import SGD

class Autoencoder(tf.keras.Model):
    """
    Autoencoder model for image reconstruction.

    Attributes:
        inp_shape: The shape of the input images (height, width, channels).
        encoded_dim: The dimension of the encoded representation.
        encoder: The encoder model.
        decoder: The decoder model.
        ae_model: The full autoencoder model.

    Methods:
        build_encoder(): Build the encoder part of the autoencoder.
        build_decoder(): Build the decoder part of the autoencoder.
        build_ae(learning_rate=0.001, momentum=0.92): Build the full autoencoder model.
        train(train_x, train_y, test_x, test_y, epochs=20, batch_size=256): Train the autoencoder model.
        plot_loss(): Plot the training loss.
    """

    def __init__(self, inp_shape=(256, 256, 1), encoded_dim=64):
        """
        Initialize the Autoencoder object.

        :param inp_shape: The shape of the input images (height, width, channels).
        :type inp_shape: tuple, optional
        :param encoded_dim: The dimension of the encoded representation.
        :type encoded_dim: int, optional
        """
        super(Autoencoder, self).__init__()
        self.inp_shape = inp_shape
        self.encoded_dim = encoded_dim

        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()
        self.ae_model = self.build_ae()

    def build_encoder(self):
        """
        Build the encoder part of the autoencoder.

        :return: The encoder model.
        :rtype: tf.keras.Model
        """
        inputs = tf.keras.Input(shape=self.inp_shape)
        x = tf.keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)

        flatten = tf.keras.layers.Flatten()(x)
        outputs = tf.keras.layers.Dense(self.encoded_dim)(flatten)

        return tf.keras.Model(inputs, outputs)

    def build_decoder(self):
        """
        Build the decoder part of the autoencoder.

        :return: The decoder model.
        :rtype: tf.keras.Model
        """
        inputs = tf.keras.Input(shape=(self.encoded_dim,))
        x = tf.keras.layers.Dense(128*128*16)(inputs)
        x = tf.keras.layers.Reshape(target_shape=(128,128,16))(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(16, (3, 3), padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.activations.relu(x)
        x = tf.keras.layers.Dropout(0.25)(x)

        outputs = tf.keras.layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
        return tf.keras.Model(inputs, outputs)

    def build_ae(self, learning_rate=0.001, momentum=0.92):
        """
        Build the full autoencoder model.

        :param learning_rate: The learning rate for the optimizer.
        :type learning_rate: float, optional
        :param momentum: The momentum value for the optimizer.
        :type momentum: float, optional
        :return: The full autoencoder model.
        :rtype: tf.keras.Model
        """
        encoder_inputs = tf.keras.Input(shape=self.inp_shape)
        encoded = self.encoder(encoder_inputs)
        decoded = self.decoder(encoded)
        ae = tf.keras.Model(encoder_inputs, decoded)
        optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=momentum)
        ae.compile(optimizer=optimizer, loss='binary_crossentropy')
        return ae

    def train(self, train_x, train_y, test_x, test_y, epochs=20, batch_size=256):
        """
        Train the autoencoder model.

        :param train_x: The input training data.
        :type train_x: np.ndarray or tf.Tensor
        :param train_y: The target training data.
        :type train_y: np.ndarray or tf.Tensor
        :param test_x: The input test data.
        :type test_x: np.ndarray or tf.Tensor
        :param test_y: The target test data.
        :type test_y: np.ndarray or tf.Tensor
        :param epochs: The number of epochs to train for.
        :type epochs: int, optional
        :param batch_size: The batch size for training.
        :type batch_size: int, optional
        """
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, mode='auto')
        his = self.ae_model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=2, validation_data=(test_x, test_y), callbacks=[early_stop], shuffle=True)
        self.his = his

    def plot_loss(self):
        """
        Plot the training loss.
        """
        plt.plot(self.his.history['loss'])
        plt.plot(self.his.history['val_loss'])
        plt.legend(['Loss', 'Val_loss'])
        plt.show()

class DataAssimilation:
    """
    A class for data assimilation using the Kalman filter.

    Attributes:
        enc_model: The encoder model used for data compression.
        dec_model: The decoder model used for data reconstruction.
        latent_dim (int): The dimension of the latent space.
        R_val (float): The coefficient of the observation error covariance matrix.
        I: The identity matrix of shape (latent_dim, latent_dim).
        R: The observation error covariance matrix.
        H: The observation operator.

    Methods:
        covariance_matrix: Compute the covariance matrix of a given dataset.
        update_prediction: Update the predicted state using the Kalman filter equations.
        KalmanGain: Compute the Kalman gain matrix.
        assimilate: Perform data assimilation using the Kalman filter.
    """

    def __init__(self, enc_model, dec_model, latent_dim=64, R_val=0.01):
        """
        Initialize the DataAssimilation object.

        :param enc_model: The encoder model used for data compression.
        :type enc_model: tf.keras.Model
        :param dec_model: The decoder model used for data reconstruction.
        :type dec_model: tf.keras.Model
        :param latent_dim: The dimension of the latent space.
        :type latent_dim: int, optional
        :param R_val: The coefficient of the observation error covariance matrix.
        :type R_val: float, optional
        """
        self.enc_model = enc_model
        self.dec_model = dec_model
        self.latent_dim = latent_dim
        self.R_val = R_val
        self.I = np.identity(latent_dim)
        self.R = R_val * self.I
        self.H = self.I 

    def covariance_matrix(self, X):
        """
        Compute the covariance matrix of a given dataset.

        :param X: The input dataset of shape (num_samples, num_features).
        :type X: np.ndarray or tf.Tensor

        :return: The covariance matrix of shape (num_features, num_features).
        :rtype: np.ndarray or tf.Tensor
        """
        means = np.array([np.mean(X, axis = 1)]).transpose()
        dev_matrix = X - means
        res = np.dot(dev_matrix, dev_matrix.transpose())/(X.shape[1]-1)
        return res

    def update_prediction(self, x, K, H, y):
        """
        Update the predicted state using the Kalman filter equations.

        :param x: The predicted state of shape (latent_dim,).
        :type x: np.ndarray or tf.Tensor
        :param K: The Kalman gain matrix of shape (latent_dim, latent_dim).
        :type K: np.ndarray or tf.Tensor
        :param H: The observation operator of shape (latent_dim, latent_dim).
        :type H: np.ndarray or tf.Tensor
        :param y: The observed state of shape (latent_dim,).
        :type y: np.ndarray or tf.Tensor

        :return: The updated state of shape (latent_dim,).
        :rtype: np.ndarray or tf.Tensor
        """
        res = x + np.dot(K, (y - np.dot(H, x)))
        return res  

    def KalmanGain(self, B, H, R):
        """
        Compute the Kalman gain matrix.

        :param B: The background error covariance matrix of shape (latent_dim, latent_dim).
        :type B: np.ndarray or tf.Tensor
        :param H: The observation operator of shape (latent_dim, latent_dim).
        :type H: np.ndarray or tf.Tensor
        :param R: The observation error covariance matrix of shape (latent_dim, latent_dim).
        :type R: np.ndarray or tf.Tensor

        :return: The Kalman gain matrix of shape (latent_dim, latent_dim).
        :rtype: np.ndarray or tf.Tensor
        """
        tempInv = inv(R + np.dot(H, np.dot(B, H.transpose())))
        res = np.dot(B, np.dot(H.transpose(), tempInv))
        return res

    def assimilate(self, bg_data, obs_data):
        """
        Perform data assimilation using the Kalman filter

        :param bg_data: The background data of shape (num_samples, height, width, channels)
        :type bg_data: np.ndarray or tf.Tensor
        :param obs_data: The observed data of shape (num_samples, height, width, channels)
        :type obs_data: np.ndarray or tf.Tensor
        :return: The updated and reconstructed data of shape (num_samples, height, width, channels)
        :rtype:  np.ndarray or tf.Tensor
        """                
        pix_1 = np.array(bg_data).shape[1]
        pix_2 = np.array(bg_data).shape[2]
        bg_compr = np.array(self.enc_model(bg_data.reshape(-1, pix_1, pix_2, 1)))
        obs_compr = np.array(self.enc_model(obs_data.reshape(-1, pix_1, pix_2, 1)))
        
        B = self.covariance_matrix(bg_compr.T)
       
        K = self.KalmanGain(B, self.H, self.R)

        updated_data_list = []
        for i in range(len(bg_compr)):
            updated_data = self.update_prediction(bg_compr[i], K, self.H, obs_compr[i]) 
            updated_data_list.append(updated_data)
        updated_data_array = np.array(updated_data_list)
        
        updated_dec = self.dec_model(updated_data_array)

        return updated_dec
    
