# Wildfire Group Project - Peshtigo

## Content
<!-- TOC -->
* [Background Briefing](#background-briefing)
* [Summary of Functionalities](#summary-of-functionalities)
* [Installation Guide](#installation-guide)
* [Files](#files)
* [User Instructions](#user-instructions)
* [Documentation](#documentation)
* [Author](#author)
<!-- TOC -->

## Background Briefing
Every year, economic losses from wildfire, propelled by the global climate change and human activities, reaches nearly 50 billion dollars. Due to the massive population growth and the rapid progress of urbanization, tackling wildfire, specifically predicting its occuring and forecasting its development both spatially and temporally, has been a critical challenge. With the assistance of Neural Networks, we are able to accomplish the tasks handed over to us. In this project, focusing on two kinds of Neural Networks: Generative Neural Networks and Recurrent Neural Networks, we take one small step for predicting wildfire but one giant leap for mankind.

## Summary of Functionalities

This repository contains three functionalities of our work. All codes are written in Python from scratch.

### Generative Neural Network - Variational AutoEncoder

* Generating new data of wildfire based on fed inputs. 
* On the Notebook, mean squared error between observation data and generated data is printed; and 4 images would be generated with 4 inputs for comparison. On input of an image at time point t=n, the prediction yields an image at t=n+1.

![VAE_origin](https://github.com/ese-msc-2022/ads-wildfire-peshtigo/assets/113195122/fd3b3e3d-278a-4121-aa3f-e3bf693763a6)
![VAE_prediction](https://github.com/ese-msc-2022/ads-wildfire-peshtigo/assets/113195122/07a27fc1-90dc-4088-bd78-e8854eb7caf9)

-----------A comparison between background information and predicted information by VAE------------
### Recurrent Neural Network - Convolutional Long Short-Term Memory

* Predicting new data of wildfire based on fed inputs.
* On the Notebook, mean squared error between observation data and generated data is printed; and 3 images would be generated with 3 inputs in which each one contains 2 timesteps.

<img width="1082" alt="Background" src="https://github.com/ese-msc-2022/ads-wildfire-peshtigo/assets/113195122/f77fa482-b1ed-476e-a599-0c9fc59d6c11">
<img width="1073" alt="Background_prediction" src="https://github.com/ese-msc-2022/ads-wildfire-peshtigo/assets/113195122/3e53dfbd-64fe-43cd-a93b-36084b47d7c7">

---------A comparison between background information and predicted information by ConvLSTM----------
                                   
### Data Assimilation 

* With a different Variational AutoEncoder, assimilating data in a latent space and decompressing the data.
* A mean square error between observation data and generated data on full space before data assimilation and a mean square error between observation data and generated data on full space after data assimilation are printed.

## Installation Guide

Before installing flood_tool it is worth ensuring your environment meets the dependencies below:

- Python > 3.9
- numpy >= 1.13.0
- pandas
- scikit-learn
- matplotlib
- pytest
- sphinx

If using conda, a new environment can be set up with the below command, after navigating to the tool directory:

```bash
conda env create -f environment.yml
```

After activating this environment,

```bash
conda activate ads
```

## Files
### .ipynb notebooks
1. Trial_and_error.ipynb: An elaboration of all previous approaches to building our LSTM model.
2. Objective_1.ipynb: Implementing all requirements from objective 1.
3. Objective_2.ipynb: Implementing all requirements from objective 2.
4. Objective_3.ipynb: Implementing all requirements from objective 3.
### .py files
The tool is delivered via a series of .py files following:

1. tools.py: Functionalities including reading data, reshaping data, generating data for models and plotting.
2. VAE.py: Functionalities including defining VAE model, calculating and visualising loss, training VAE model, testing VAE model.
3. DA.py: Functionalities including defining linear AE model, training linear AE model, testing linear AE model, compressing data, assimilating data and decompressing data.
4. LSTM.py: Functionalities including defining LSTM model, calculating and visualising loss, training LSTM model, testing LSTM model.
5. test: Multiple test.py files for testing all the other functions in the project. 

### Saved models
For ConvLSTM model, a trained version is available from the following link:<br />
[https://drive.google.com/file/d/1-8-6CLHSkC6jtTe4HbBFguD5pNs8blG_/view?usp=share_link]<br />
For VAE model, a trained version is available from the following link:<br />
[https://drive.google.com/file/d/1G7VGKmlk660BPeP1l2VHZj8wl_H01jiD/view?usp=sharing]

## User instructions
#### Variational AutoEncoder
```
import VAE
import tools
```
* Read data and reshape data into TensorFlow
* Train and test VAE model
* Generate result

#### Convolutional Long Short-Term Memory
```
import LSTM
import tools
```
* Read data and reshape data into TensorFlow
* Train and test ConvLSTM model
* Generate result

#### Data Assimilation
```
import DA
import tools
```
* Read data and reshape data into TensorFlow
* Train and test Linear VAE model
* Data compression
* Data assimilation
* Data decompression and generate result

### Testing

The tool includes several tests, which you can use to check its operation on your system. With [pytest](https://doc.pytest.org/en/latest) installed, these can be run with

```bash
pytest test_LSTM.py
```

Additional more specific tests have been saved in the test directory. These can be run where required to check different parts of the tool functionality individually.

- `test_VAE.py` 
- `test_LSTM.py`
- `test_DA`
- `test_tools`

## Documentation
In the [ads-wildfire-peshtigo.pdf](#ads-wildfire-peshtigo.pdf), it contains detailed documentation for the package, including examples and details of all functions and inputs.

## Author
Team Peshtigo(Alphabetical):<br />
Fan Feng<br />
Ligen Cai<br />
Luisina Canto<br />
Leyang Zhang<br />
Manya Sinha<br />
Wenxin Ran<br />
Yulin Zhuo<br />
Zongyang Gao<br />
