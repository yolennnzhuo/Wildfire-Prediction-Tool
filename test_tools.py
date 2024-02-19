import numpy as np
import pytest
import sys
sys.path.insert(0,'..')
from tools import reshape_2d, select_samples, pred_n_steps, plot_data, generate_lstm_input, reshape_background

def test_reshape_2d():
    dd = np.random.rand(10, 5, 5)
    dd_new = reshape_2d(dd)
    assert dd_new.shape == (10, 25)

def test_select_samples():
    dd = np.random.rand(1000, 5)
    dd_new = select_samples(dd, start=0, step=100)
    assert dd_new.shape == (990, 5)

def test_generate_lstm_input():
    data = np.random.rand(12500, 256, 256)
    num_batches = 125

    input_data, output_data = generate_lstm_input(data, num_batches)
    assert input_data.shape == (9875, 2, 256, 256, 1)
    assert output_data.shape == (9875, 256, 256, 1)

def test_reshape_background():
    background = np.random.rand(5, 256, 256)
    input_data, output_data = reshape_background(background)
    assert input_data.shape == (3, 2, 256, 256, 1)
    assert output_data.shape == (3, 256, 256, 1)

def test_plot_data():
    data = np.random.rand(10, 256, 256)
    ts = [0, 1, 2]
    titles = ["t=0", "t=1", "t=2"]

    # We are testing the function call only because it does not have a return statement
    # We assume that if it does not raise an exception, it has passed the test
    try:
        plot_data(data, ts, titles)
    except Exception as e:
        pytest.fail(f"plot_data raised exception {e}")

def test_pred_n_steps():
    from keras.models import Sequential
    from keras.layers import Dense

    # Creating a simple model for testing
    model = Sequential()
    model.add(Dense(units=64, activation='relu', input_shape=(10,)))
    model.add(Dense(units=10, activation='softmax'))
    model.compile(optimizer='sgd', loss='categorical_crossentropy')

    dd = np.random.rand(10, 10)
    steps = 2

    assert pred_n_steps(model, dd, steps).shape == (10,10) 
