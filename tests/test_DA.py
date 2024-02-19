import pytest
import numpy as np
import tensorflow as tf
import sys
sys.path.insert(0,'..')
from DA import Autoencoder, DataAssimilation

@pytest.fixture
def autoencoder_instance():
    return Autoencoder()

@pytest.fixture
def data_assimilation_instance(autoencoder_instance):
    enc_model = autoencoder_instance.encoder
    dec_model = autoencoder_instance.decoder
    return DataAssimilation(enc_model, dec_model)

def test_autoencoder_build_encoder(autoencoder_instance):
    """
    Test function to validate the build_encoder method of the Autoencoder class.

    Args:
        autoencoder_instance (Autoencoder): An instance of the Autoencoder class.

    Raises:
        AssertionError: If the output of build_encoder is not an instance of tf.keras.Model.

    Example:
        >>> autoencoder = Autoencoder()
        >>> test_autoencoder_build_encoder(autoencoder)
    """
    encoder = autoencoder_instance.build_encoder()
    assert isinstance(encoder, tf.keras.Model)

def test_autoencoder_build_decoder(autoencoder_instance):
    """
    Test function to validate the build_decoder method of the Autoencoder class.

    Args:
        autoencoder_instance (Autoencoder): An instance of the Autoencoder class.

    Raises:
        AssertionError: If the output of build_decoder is not an instance of tf.keras.Model.

    Example:
        >>> autoencoder = Autoencoder()
        >>> test_autoencoder_build_decoder(autoencoder)
    """
    decoder = autoencoder_instance.build_decoder()
    assert isinstance(decoder, tf.keras.Model)

def test_autoencoder_build_ae(autoencoder_instance):
    """
    Test function to validate the build_ae method of the Autoencoder class.

    Args:
        autoencoder_instance (Autoencoder): An instance of the Autoencoder class.

    Raises:
        AssertionError: If the output of build_ae is not an instance of tf.keras.Model.

    Example:
        >>> autoencoder = Autoencoder()
        >>> test_autoencoder_build_ae(autoencoder)
    """
    ae_model = autoencoder_instance.build_ae()
    assert isinstance(ae_model, tf.keras.Model)

def test_autoencoder_train(autoencoder_instance):
    """
    Test function to validate the train method of the Autoencoder class.

    Args:
        autoencoder_instance (Autoencoder): An instance of the Autoencoder class.

    Raises:
        AssertionError: If the Autoencoder instance does not have the 'his' attribute after training.

    Example:
        >>> autoencoder = Autoencoder()
        >>> test_autoencoder_train(autoencoder)
    """
    train_x = np.random.randn(100, 256, 256, 1)
    train_y = np.random.randn(100, 256, 256, 1)
    test_x = np.random.randn(20, 256, 256, 1)
    test_y = np.random.randn(20, 256, 256, 1)
    autoencoder_instance.train(train_x, train_y, test_x, test_y)
    assert hasattr(autoencoder_instance, 'his')

def test_data_assimilation_covariance_matrix(data_assimilation_instance):
    """
    Test function to validate the covariance_matrix method of the DataAssimilation class.

    Args:
        data_assimilation_instance (DataAssimilation): An instance of the DataAssimilation class.

    Raises:
        AssertionError: If the shape of the covariance matrix is not as expected.

    Example:
        >>> enc_model = Autoencoder().encoder
        >>> dec_model = Autoencoder().decoder
        >>> data_assimilation = DataAssimilation(enc_model, dec_model)
        >>> test_data_assimilation_covariance_matrix(data_assimilation)
    """
    X = np.random.randn(100, 100)
    cov_matrix = data_assimilation_instance.covariance_matrix(X)
    assert cov_matrix.shape == (X.shape[0], X.shape[0])



def test_data_assimilation_update_prediction(data_assimilation_instance):
    """
    Test function to validate the update_prediction method of the DataAssimilation class.

    Args:
        data_assimilation_instance (DataAssimilation): An instance of the DataAssimilation class.

    Raises:
        AssertionError: If the shape of the updated prediction is not as expected.

    Example:
        >>> enc_model = Autoencoder().encoder
        >>> dec_model = Autoencoder().decoder
        >>> data_assimilation = DataAssimilation(enc_model, dec_model)
        >>> x = np.random.randn(64)
        >>> K = np.random.randn(64, 64)
        >>> H = np.eye(64)
        >>> y = np.random.randn(64)
        >>> test_data_assimilation_update_prediction(data_assimilation)
    """
    x = np.random.randn(data_assimilation_instance.latent_dim)
    K = np.random.randn(data_assimilation_instance.latent_dim, data_assimilation_instance.latent_dim)
    H = np.eye(data_assimilation_instance.latent_dim)
    y = np.random.randn(data_assimilation_instance.latent_dim)

    updated_prediction = data_assimilation_instance.update_prediction(x, K, H, y)
    assert updated_prediction.shape == (data_assimilation_instance.latent_dim,)

def test_data_assimilation_KalmanGain(data_assimilation_instance):
    """
    Test function to validate the KalmanGain method of the DataAssimilation class.

    Args:
        data_assimilation_instance (DataAssimilation): An instance of the DataAssimilation class.

    Raises:
        AssertionError: If the shape of the Kalman gain matrix is not as expected.

    Example:
        >>> enc_model = Autoencoder().encoder
        >>> dec_model = Autoencoder().decoder
        >>> data_assimilation = DataAssimilation(enc_model, dec_model)
        >>> B = np.random.randn(64, 64)
        >>> H = np.eye(64)
        >>> R = np.eye(64) * 0.01
        >>> test_data_assimilation_KalmanGain(data_assimilation)
    """
    B = np.random.randn(data_assimilation_instance.latent_dim, data_assimilation_instance.latent_dim)
    H = np.eye(data_assimilation_instance.latent_dim)
    R = np.eye(data_assimilation_instance.latent_dim) * data_assimilation_instance.R_val

    Kalman_gain = data_assimilation_instance.KalmanGain(B, H, R)
    assert Kalman_gain.shape == (data_assimilation_instance.latent_dim, data_assimilation_instance.latent_dim)

def test_data_assimilation_assimilate(data_assimilation_instance):
    """
    Test function to validate the assimilate method of the DataAssimilation class.

    Args:
        data_assimilation_instance (DataAssimilation): An instance of the DataAssimilation class.

    Raises:
        AssertionError: If the shape of the updated decoded data is not as expected.

    Example:
        >>> enc_model = Autoencoder().encoder
        >>> dec_model = Autoencoder().decoder
        >>> data_assimilation = DataAssimilation(enc_model, dec_model)
        >>> bg_data = np.random.randn(10, 256, 256)
        >>> obs_data = np.random.randn(10, 256, 256)
       """
    bg_data = np.random.randn(10, 256, 256)
    obs_data = np.random.randn(10, 256, 256)

    updated_decoded_data = data_assimilation_instance.assimilate(bg_data, obs_data)
    assert updated_decoded_data.shape == (10, 256, 256, 1)
