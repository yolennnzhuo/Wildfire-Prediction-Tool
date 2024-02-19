import pytest
import numpy as np
import sys
sys.path.insert(0,'..')
import LSTM
import tensorflow as tf

class TestLSTMModel():
    @pytest.fixture(autouse=True)
    def setUp(self):
        '''
        Set up the model and variables required for testing
        '''
        self.model = LSTM.ConvLSTMModel(convlstm_input_shape=(3, 256, 256, 1))
        self.x_train = np.random.rand(10, 3, 256, 256, 1)
        self.y_train = np.random.rand(10, 256, 256, 1)
        self.x_test = np.random.rand(2, 3, 256, 256, 1)
        self.y_test = np.random.rand(2, 256, 256, 1)

    def test_build_conv_lstm(self):
        '''
        Test the 'build_conv_lstm' function by checking the type of the model
        '''
        conv_lstm = self.model.build_conv_lstm()
        assert isinstance(conv_lstm, tf.keras.models.Sequential)

    def test_call(self):
        '''
        Test the 'call' function by checking the shape of the model output
        '''
        output = self.model(self.x_train)
        assert output.shape == self.y_train.shape

    def test_train(self):
        '''
        Test the 'train' function by checking the shape of the model output
        '''
        self.model.train(self.x_train, self.y_train, self.x_test, self.y_test, epochs=1)
        assert len(self.model.his.history['loss']) == 1

    def test_plot_loss(self):
        '''
        Test the 'plot_loss' function by checking if it can run successfully
        '''
        try:
            # Train the model for plotting the loss
            self.model.train(self.x_train, self.y_train, self.x_test, self.y_test, epochs=1)
            # Call the plot function
            self.model.plot_loss()

        except Exception as e:
            assert False, f"plot_loss() raised an exception: {e}"

