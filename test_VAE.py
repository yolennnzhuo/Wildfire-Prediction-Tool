# import VAE
import sys
sys.path.insert(0,'..')
from VAE import VAE
import pytest
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

@pytest.fixture
def vae_instance():
    """Fixture for creating an instance of the VAE class.

    Returns:
        VAE: An instance of the VAE class.
    """
    return VAE(latent_dim=128, input_dim=256*256*1)


def test_vae_inputs():
    """Test dimensions and properties of the VAE inputs.

    Raises:
        AssertionError: If any of the dimension checks fail or if the properties of inputs are invalid.
    """
    latent_dim = 128
    input_dim = 256 * 256 * 1

    # Check dimensions are not 0
    assert isinstance(latent_dim, int) and latent_dim != 0
    assert isinstance(input_dim, int) and input_dim != 0

    try:
        # only pass in the parameters that the VAE constructor can accept
        vae = VAE(latent_dim, input_dim)
    except Exception as e:
        pytest.fail(f"Failed to create VAE: {e}")


def test_build_encoder(vae_instance):
    """Validates the construction of the encoder.

    Args:
        vae_instance (VAE): An instance of the VAE class.

    Raises:
        AssertionError: If the encoder is not an instance of `tf.keras.Model`,
            if the number of encoder inputs is not equal to 1,
            or if the number of encoder outputs is not equal to 3.
    """
    encoder = vae_instance.build_encoder()
    assert isinstance(encoder, tf.keras.Model)
    assert len(encoder.inputs) == 1
    assert len(encoder.outputs) == 3


def test_build_decoder(vae_instance):
    """Validates the construction of the decoder.

    Args:
        vae_instance (VAE): An instance of the VAE class.

    Raises:
        AssertionError: If the decoder is not an instance of `tf.keras.Model`,
            if the number of decoder inputs is not equal to 1,
            or if the number of decoder outputs is not equal to 1.
    """
    decoder = vae_instance.build_decoder()
    assert isinstance(decoder, tf.keras.Model)
    assert len(decoder.inputs) == 1
    assert len(decoder.outputs) == 1


def test_build_vae(vae_instance):
    """Validates the construction of the VAE.

    Args:
        vae_instance (VAE): An instance of the VAE class.

    Raises:
        AssertionError: If the VAE is not an instance of `tf.keras.Model`,
            if the number of VAE inputs is not equal to 1,
            or if the number of VAE outputs is not equal to 1.
    """
    vae = vae_instance.build_vae()
    assert isinstance(vae, tf.keras.Model)
    assert len(vae.inputs) == 1
    assert len(vae.outputs) == 1


def test_sampling(vae_instance):
    """Validates the sampling method in the VAE.

    Args:
        vae_instance (VAE): An instance of the VAE class.

    Raises:
        AssertionError: If the sampling method does not return the expected shape or dtype.
    """
    # Create dummy inputs
    dummy_args = (tf.random.normal(shape=(10, vae_instance.latent_dim)),
                  tf.random.normal(shape=(10, vae_instance.latent_dim)))

    # Call the sampling method
    sampled_output = vae_instance.sampling(dummy_args)

    # Check the shape and dtype of the sampled output
    expected_shape = (10, vae_instance.latent_dim)
    expected_dtype = tf.float32
    assert sampled_output.shape == expected_shape, f"Expected shape: {expected_shape}, Actual shape: {sampled_output.shape}"
    assert sampled_output.dtype == expected_dtype, f"Expected dtype: {expected_dtype}, Actual dtype: {sampled_output.dtype}"


def test_loss_function(vae_instance):
    """Validates the loss_function method in the VAE.

    Args:
        vae_instance (VAE): An instance of the VAE class.

    Raises:
        AssertionError: If the loss function does not return the expected shape or dtype.
    """
    # Create dummy inputs and outputs
    inputs = tf.random.normal(shape=(10, vae_instance.input_dim))
    outputs = tf.random.normal(shape=(10, vae_instance.input_dim))
    z_mean = tf.random.normal(shape=(10, vae_instance.latent_dim))
    z_log_var = tf.random.normal(shape=(10, vae_instance.latent_dim))

    # Call the loss_function method
    loss = vae_instance.loss_function(inputs, outputs, z_mean, z_log_var)

    # Check the shape and dtype of the loss
    expected_shape = ()
    expected_dtype = tf.float32
    assert loss.shape == expected_shape, f"Expected shape: {expected_shape}, Actual shape: {loss.shape}"
    assert loss.dtype == expected_dtype, f"Expected dtype: {expected_dtype}, Actual dtype: {loss.dtype}"


@pytest.fixture
def x_train_y_train():
    """Create a sample training set."""
    return np.random.rand(100, 256 * 256 * 1), np.random.rand(100, 256 * 256 * 1)


@pytest.fixture
def x_test_y_test():
    """Create a sample validation set."""
    return np.random.rand(20, 256 * 256 * 1), np.random.rand(20, 256 * 256 * 1)


@pytest.fixture
def batch_size():
    """Fixture for batch size."""
    return 64


@pytest.fixture
def epochs():
    """Fixture for number of epochs."""
    return 10


def test_train(vae_instance, x_train_y_train, x_test_y_test, batch_size, epochs):
    """Test the training function of the VAE class.

    Args:
        vae_instance (VAE): An instance of the VAE class.
        x_train_y_train (tuple): Training data as a tuple of input and output arrays.
        x_test_y_test (tuple): Validation data as a tuple of input and output arrays.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.

    Raises:
        AssertionError: If training fails with an error.
    """
    x_train, y_train = x_train_y_train
    x_test, y_test = x_test_y_test
    try:
        vae_instance.train(x_train, y_train, batch_size, epochs, x_test, y_test)
    except Exception as e:
        pytest.fail(f"Training failed with error: {e}")


def plot_loss(vae_instance):
    """Plot and save the loss plot for the VAE instance.

    Args:
        vae_instance (VAE): An instance of the VAE class.
    """
    vae_instance.plot_loss()
    plt.savefig("loss_plot.png")


def test_plot_loss(vae_instance, x_train_y_train, x_test_y_test, batch_size, epochs):
    """Test the plot_loss function for the VAE instance.

    Args:
        vae_instance (VAE): An instance of the VAE class.
        x_train_y_train (tuple): Training data as a tuple of input and output arrays.
        x_test_y_test (tuple): Validation data as a tuple of input and output arrays.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.

    Raises:
        AssertionError: If the loss plot file is not created.
    """
    x_train, y_train = x_train_y_train
    x_test, y_test = x_test_y_test
    vae_instance.train(x_train, y_train, batch_size, epochs, x_test, y_test)
    plot_loss(vae_instance)
    assert os.path.isfile("loss_plot.png"), "Loss plot file is not created."
