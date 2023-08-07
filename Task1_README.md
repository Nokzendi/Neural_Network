# Prerequisite Task 1

Streamlit App for a customizable Neural Network

## Table of Contents

- [Description](#description)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)

## Description

This is a Streamlit App that a user can use to play around with Neural Networks. It has custom inputs for the user to adjust the architecture of the Neural Network as well as a list of datasets to train the model. The app shows the decision boundary on the contour plot as the probabilities along with the datapoints used in the model. 

The settings for the neural network are located in the sidebar while the settings for the datasets are located in the main page. The main page is where the output and plots are shown as well.

## Features

- 5 different types of dataset
    - Concentric Circle
    - Spiral
    - XOR 
    - Two Gauss
    - Moons
- Can select the number of datapoints in each dataset
- Can adjust the noise in each dataset
- Can select the ratio of train test split
- Can select basis transformation
    - x^2
    - y^2
    - x * y
    - sin(x)
    - sin(y)
    - exp(-x^2)
    - exp(-y^2)
- Can adjust the number of hidden layers
- Can select the learning rate
- Can adjust the number of epochs to run the model
- Can select the number of output nodes (Kept at 2 for this demo)
- Can select whether to use MC Dropouts
- Can adjust the number of neurons in each hidden layer

## Installation

Install the necessary libraries used using pip. Once installed, go to your terminal and 'cd' into the directory that has the 'task1.py' file. Run the following comamnd:

```bash
streamlit run task1.py
```

## Usage
Once you run the command above, the app should open up in your browser. You can also find the hosted app in this [link](https://neuralnetwork-wpacpmpr53nx267dgcjsr9.streamlit.app/). Enjoy.
