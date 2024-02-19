import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def reshape_2d(dd):
    """
    This function reshape a 3d dataset to 2d kept first dimension unchanged

    :param dd: 3d dataset
    :type dd: np.ndarray or tf.Tensor
    :return: reshaped 2d dataset
    :rtype: np.ndarray or tf.Tensor
    """
    dd_new = dd.reshape(-1, dd.shape[1]*dd.shape[2])
    return dd_new

def select_samples(dd, start=0, step=100):
    """
    This function selects samples from a training set by taking every 'step'-th sample.

    :param dd: 2d or 3d dataset
    :type dd: np.ndarray or tf.Tensor
    :param start: the first element to remove
    :type start: int
    :param step: the step size for sample selection
    :type step: int
    :return: dataset after selection
    :rtype: np.ndarray or tf.Tensor
    """   
    indices = np.array(range(start, dd.shape[0], step))
    mask = np.ones(dd.shape[0], bool)
    mask[indices] = False
    dd_new = dd[mask, :].reshape(-1, dd.shape[1])
    return dd_new

def plot_data(data, ts, titles):
    """
    Creates a figure and a set of subplots.

    :param data: dataset 
    :type data: np.ndarray or tf.Tensor
    :param ts: timesteps to plot
    :type ts: list of int
    :param titles: titles for each subplot
    :type titles: list of str
    """
    fig, axes = plt.subplots(1, len(ts), figsize=(20, 8))
    for t, title, ax in zip(ts, titles, axes.ravel()):
        im = ax.imshow(data[t])
        ax.set_title(title)
    plt.show()


def pred_n_steps(model, dd, steps):
    """
    Predict behaviors at (t+n) for given model

    :param model: model used to predict
    :type model: tf.keras.Model
    :param dd: data used for prediction:
    :type dd: np.ndarray or tf.Tensor
    :param steps: n timesteps later
    :type steps: int
    :return: predicted data at timestep (t+n)
    :rtype: np.ndarray or tf.Tensor
    """
    y = np.copy(dd)
    for i in range(steps):
      y = model.predict(y)
    return y

def generate_lstm_input(data, num_batches):
    """
    Generates input data set for an LSTM model from the given data.

    :param data: The original dataset from which the inputs are generated. 
    :type data: np.ndarray or tf.Tensor
    :param num_batches: The number of batches to be generated.
    :type num_batches: int
    :return: The input data for the model and the target data.
    :rtype: tuple of np.ndarray
    """
    lstm_input = []
    
    for n in range(num_batches):
        # Select a segment(n_th segment) of the input data
        data_segment = data[n*100:(n+1)*100,:,:]

        new_batch = []
        for i in range(79):
            selected_train = data_segment[i:21+i:10, :, :]  # Select every 10th time step before time step 100
            new_batch.append(selected_train)

        # Reshape the data to be suitable for input to an LSTM model
        new_batch = np.reshape(new_batch, (-1, 3, 256, 256, 1))
        
        # Append the prepared sample to the lstm_input list
        lstm_input.append(new_batch)
    
    # Convert the list to a numpy array for further processing
    lstm_input = np.array(lstm_input)
    
    # Reshape the array to have a uniform shape for all samples
    lstm_input = np.reshape(lstm_input, (-1, 3, 256, 256, 1))
    
    return lstm_input[:,:2,:,:], lstm_input[:,-1,:,:]


def reshape_background(background):
    """
    Reshape the background data for LSTM model

    :param background: Input data for LSTM model
    :type background: np.ndarray or tf.Tensor
    :return: Reshaped input data and target data
    :rtype: np.ndarray or tf.Tensor
    """
    back_pred = []
    for i in range(3):
        selected_back = background[i:i+3, :, :]  # Select the first 3 time step to predict
        back_pred.append(selected_back)
    background_reshape = np.reshape(back_pred, (-1, 3, 256, 256, 1))

    return background_reshape[:,:2,:,:], background_reshape[:,-1,:,:]
