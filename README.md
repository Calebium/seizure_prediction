# Seizure prediction using real-time EEG and LSTM

## Introduction

This is an LSTM (long short-term memory) based classification model that takes continuous, multi-channel EEG signal for the prediction of serizures in patients with epilepsy. The training data used was collected from 22 patients from the Boston Children's Hospital using the international 10-20 system, and the training labels were annotated by medical professionals (https://physionet.org/content/chbmit/1.0.0/).

![Image of EEG recording with an annotated seizure](https://github.com/Calebium/seizure_prediction/tree/main/Images/sample_seizure.png)

## Code description

The code was originally written in Jupyter notebooks, but can be run either interactively (\*.ipynb files) or from the command line (\*.py files). The code was written to detect automatically whether it is being run interactively or in batch mode, and takes input arguments from the respective environments (from within the notebook or from the command line). The two main scripts, `scrTrainLSTM.py` and `scrTestLSTM.py`, are used for training and testing the model, respectively. The remaining files are all library modules utilized by these two scripts. More detailed descriptions below.

### Training and testing scripts (`scrTrainLSTM.py` and `scrTestLSTM.py`)

The main input to the both the training and test scripts is a CSV file that points to one or more EEG recordings (in .edf format) as training or test data. Various hyperparameters can also be specified that relates to the following areas:

* Model architecture (e.g. number of layers, number of predicted classes, and dropout rate)

* Training parameters (e.g. optimizer, learning rate, class weights, number of epochs)

* Data set split (e.g. split ratio between training and validation sets)

* EEG-specific parameters (e.g. sampling rate, scanning window size, normalization strategy)

For the test script, the main arguments are the location of the trained model and the CSV file for the test data.

### Library modules

The library modules are divided into the following categories:

* Data I/O (`libDataIO.py`) - for reading in EEG (.edf) files and the corresponding seizure labels, and breaking up the EEG recordings into small segments as extracted by a sliding window before being fed into the model for training or testing

* Model architecture (`libModelLSTM.py`) - for constructing, loading, or saving an LSTM model

* Other utility functions (`libUtils.py`) - for all other reusable functions shared between the training and test scripts

### Example

Given a valid CSV file that points to existing EEG (.edf) files, the following script can be run to start a training:

`bash runTrainLSTM.sh`
