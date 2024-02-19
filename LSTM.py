import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class ConvLSTMModel(tf.keras.Model):
    """
    A class used to represent a Convolutional LSTM Model.

    Attributes:
        convlstm_input_shape: Shape of the input data for the ConvLSTM layer.
        conv_lstm_layer: Convolutional LSTM layer created using the build_conv_lstm() method.

    Methods:
        build_conv_lstm(): Build a Convolutional LSTM (ConvLSTM) model with specified parameters.
        call(inputs): Run the ConvLSTM model on the provided inputs.
        train(x_train, y_train, x_test, y_test, lr = 0.01, batch_size=64, epochs=50): Train the ConvLSTM model with the provided training data.
        plot_loss(): Plot the model's training and validation loss over epochs.
    """

    def __init__(self, convlstm_input_shape=(None, None, None, None)):
        super(ConvLSTMModel, self).__init__()
        """
        Initialize the ConvLSTMModel class.

        :param convlstm_input_shape: The shape of the input data for the ConvLSTM layer.
        :type onvlstm_input_shape: tuple
        """
        self.convlstm_input_shape = convlstm_input_shape
        self.conv_lstm_layer = self.build_conv_lstm()
        
    def build_conv_lstm(self):
        """
        Build a Convolutional LSTM (ConvLSTM) model.

        :return: A ConvLSTM model with a single ConvLSTM2D layer.
        :rtype: tf.keras.models.Sequential
        """
        model = tf.keras.models.Sequential()
        model.add(layers.ConvLSTM2D(filters=1, kernel_size=(3, 3), input_shape=self.convlstm_input_shape, padding='same', activation='sigmoid', return_sequences=False))
        return model

    def call(self, inputs):
        """
        Execute the ConvLSTM model on the provided inputs.

        :param inputs: The input data or features to be passed through the ConvLSTM model.
        :type inputs: np.ndarray or tf.Tensor
        :return: The output of the ConvLSTM model.
        :rtype: tf.Tensor
        """
        return self.conv_lstm_layer(inputs)
    
    def train(self, x_train, y_train, x_test, y_test, lr=0.01, batch_size=64, epochs=50):
        """
        Train the ConvLSTM model.

        :param x_train: The input training data.
        :type x_train: np.ndarray or tf.Tensor
        :param y_train: The target training data.
        :type y_train: np.ndarray or tf.Tensor
        :param x_test: The input test data.
        :type x_test: np.ndarray or tf.Tensor
        :param y_test: The target test data.
        :type y_test: np.ndarray or tf.Tensor
        :param epochs: The number of epochs to train for.
        :type epochs: int, optional
        :param batch_size: The batch size for training.
        :type batch_size: int, optional
        :param lr: The learning rate for the optimizer.
        :type lr: float, optional
        """
        # Create an Adam optimizer with the given learning rate
        opt = Adam(learning_rate = lr)
        # Compile the model
        self.compile(optimizer=opt, loss='mean_squared_error')
        
        # Train the model
        his = self.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_data = (x_test, y_test), shuffle = True)
        self.his = his
    
    def plot_loss(self):
        """
        Plot the model's training and validation loss over epochs.
        """
        plt.plot(self.his.history['loss'])
        plt.plot(self.his.history['val_loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.legend(['Loss', 'Val_loss'])
        plt.show()