import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.layers import Dropout
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import seaborn as sns
import datasets as d
import pandas as pd

np.random.seed(42)

## Constants
LEARNING_RATE_OPTIONS = [0.00001, 0.0001, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]
DATASET_SIZE_OPTIONS = [100, 500, 1000]
BASIS_TRANSFOMATION = ['x^2', 'y^2', 'x*y', 'Sin(x)', 'Sin(y)', 'exp(-x^2)', 'exp(-y^2)']

st.set_page_config(layout = 'wide')
st.sidebar.title('Neural Network Settings')

class MCDropout(Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)
                            
def plot_points(points):
    """
    Function to plot the datapoints on a 2D plot using matplotlib library based on the labels.

    Parameters
    ----------
    points : 1D_array_like
        List of 'DataPoint' objects that describe the coordinates and the output label of the datapoint
    """
    fig, ax = plt.subplots()

    positive_points = [p for p in points if p.label == 1]
    negative_points = [p for p in points if p.label == 0]

    positive_x = [p.x for p in positive_points]
    positive_y = [p.y for p in positive_points]
    negative_x = [p.x for p in negative_points]
    negative_y = [p.y for p in negative_points]

    ax.scatter(positive_x, positive_y, c='orange', label='1')
    ax.scatter(negative_x, negative_y, c='blue', label='0')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.legend()
    ax.set_title('Positive and Negative Labeled Points')
    ax.grid(True)
    st.pyplot(fig)

## Selecting Datasets
def select_dataset():
    """
    Function to select the dataset type based on user inputs

    Returns
    -------
    points : 1D_array_like
        List of 'DataPoint' objects that describe the coordinates and the output label of the datapoint
    """
    dataset_type = st.radio('', ('Concentric Circle', 'Spiral', 'XOR Data', 'Gauss', 'Moons'))
    dataset_size = st.selectbox('Number of datapoints', DATASET_SIZE_OPTIONS, index = DATASET_SIZE_OPTIONS.index(100))
    dataset_noise = st.slider('Noise', min_value = 0.0, max_value = 0.1, value = 0.01, step = 0.01)

    if(dataset_type == 'Concentric Circle'):
        points = d.classify_circle_data(dataset_size, dataset_noise)
    elif(dataset_type == 'Spiral'):
        points = d.classify_spiral_data(dataset_size, dataset_noise)
    elif(dataset_type == 'XOR Data'):
        points = d.classify_xor_data(dataset_size, dataset_noise)
    elif(dataset_type == 'Gauss'):
        points = d.classify_two_gauss_data(dataset_size, dataset_noise)
    elif(dataset_type == 'Moons'):
        points = d.classify_moons_data(dataset_size, dataset_noise)
    return points

## Separating training and testing datapoints
def get_training_testing(points, labels, split):
    """
    Function that creates a train test split from the dataset

    Parameters
    ----------
    points : 2D_array_like
        Numpy matrix of datapoints. Each row represents a different coordinate and each column represents a different basis
    labels : 1D_array_like
        Numpy array of intergers that holds the output labels of the coordinates stored in 'points'
    split : float
        Number indicating the ratio of testing to training split
    """
    X_train, X_test, y_train, y_test = train_test_split(points, labels, test_size=split, random_state=42)
    return X_train, X_test, y_train, y_test

## Separating datapoints and labels
def get_points_labels(points):
    """
    Function that separates the x and y coordinates from its corresponding labels

    Parameters
    ----------
    points : 1D_array_like
        List of 'DataPoint' objects that describe the coordinates and the output label of the datapoint
    
    Returns
    -------
    raw_points : 2D_array_like
        Matrix that holds the x and y coordinates in separate columns
    labels : 1D_array_like
        List of intergers that holds the output labels of the coordinates stored in 'raw_points'
    """
    raw_points = []
    labels = []
    for p in points:
        raw_points.append([p.x, p.y])
        labels.append(p.label)
    return np.array(raw_points), np.array(labels)

## Creating Neural Network
def create_model(num_layers, lr, num_classes, mc_dropout):
    """
    Function that creates a model based on user inputs

    Parameters
    ----------
    num_layers : int
        Number of hidden layers in the neural network
    lr : float
        Learning rate of the neural network
    num_classes : int
        Number of outupt classes
    mc_dropout : bool
        Whether to use MC Droputs in the neural network

    Returns
    -------
    model : Neural Network model created using 'Tensorflow Keras' library
    """
    model = keras.Sequential()
    for i in range(num_layers):
        with st.sidebar:
            num_neurons = st.number_input(f'Number of neurons in hidden layer {i+1}', min_value = 1, step = 1, value = 128)
        model.add(keras.layers.Dense(num_neurons, activation = 'relu'))
        if mc_dropout:
            model.add(MCDropout(0.4))
    model.add(keras.layers.Dense(num_classes, activation = 'softmax'))

    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model

## Training model
def train_model(model, train_points, train_labels, epochs, test_points, test_labels):
    """
    Function to train the model

    Parameters
    ----------
    model : Neural Network model
        The neural network model created using 'Tensorflow Keras' library
    train_points : 2D_array_like
        Numpy array that holds the input values. Matrix of coordinates with a different datapoint in each row and 
        different basis in each column
    train_labels : 1D_array_like
        Numpy array that holds the output values/classes
    epochs : int
        Number of epochs to train our model
    test_points : 2D_array_like
        Numpy array that holds the input values same as train_points. Used for validation while training
    test_labels : 1D_array_like
        Numpy array that holds the output values/classes

    Returns
    -------
    history : History of the model from 'Tensorflow Keras' library
    """
    history = model.fit(train_points, train_labels, epochs=epochs, validation_data = (test_points, test_labels))
    return history

## Plotting contour plots and predictions
def plot_contour(model, points, labels, basis_transformation, mc_dropout):
    """
    Function to plot the contour on the probabilities

    Parameters
    ----------
    model : Neural Network model
        The neural network model created using 'Tensorflow Keras' library
    points : 2D_array_like
        Numpy array that holds the input values. Matrix of coordinates with a different datapoint in each row and 
        different basis in each column
    labels : 1D_array_like
        Numpy array that holds the output values/classes
    basis_transformation : 1D_array_like
        List of strings that represents which basis transformation to perform on the datapoints
    mc_dropout : bool
        Whether to use MC Droputs in the neural network
    """
    # Generate a grid of points to cover the entire input space
    x_min, x_max = -6, 6
    y_min, y_max = -6, 6
    step = 0.1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    transformed_grid_points = transform_points(grid_points, basis_transformation)

    # Use the trained neural network to make predictions (probabilities) on the grid
    probabilities = model.predict(transformed_grid_points)

    # Take an average of 5 when mc_droput is enabled
    if mc_dropout:
        for i in range(4):
            prob = model.predict(transformed_grid_points)
            probabilities += prob
        probabilities /= 5
        

    # Reshape probabilities to match the grid shape for contour plot
    probabilities = probabilities[:, 1]
    probabilities = probabilities.reshape(xx.shape)

    # Plot the contour plot with probabilities
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, probabilities, alpha=0.8, cmap='viridis')

    # Separating the x and y coordinates
    x_points = points[:, 0]
    y_points = points[:, 1]
    plot_labels = labels

    # Scatter plot the actual data points with customized labels
    sns.scatterplot(x=x_points, y=y_points, hue=plot_labels, palette=['blue', 'orange'], s=50)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Contour Plot with Probabilities')
    plt.grid(True)

    # Convert the plot to a Streamlit-compatible format using st.pyplot()
    st.pyplot(fig)

## Training Results
def model_results(history):
    """
    Function to show the accuracy and loss results of the model

    Parameters
    ----------
    history : History of the model from 'Tensorflow Keras' library
    """
    train_accuracy = history.history['accuracy'][-1]
    train_loss = history.history['loss'][-1]
    train_output = {"Training Accuracy" : train_accuracy, "Training Loss" : train_loss}
    st.dataframe(train_output)

    test_accuracy = history.history['val_accuracy'][-1]
    test_loss = history.history['val_loss'][-1]
    test_output = {"Validation Accuracy" : test_accuracy, "Validation Loss" : test_loss}
    st.dataframe(test_output)

## Transform basis of dataset
def transform_points(points, basis):
    """
    Function to perform the basis transformation

    Parameters
    ----------
    points : 2D_array_like
        Numpy array that holds the input values. Matrix of coordinates with a different datapoint in each row and 
        different basis in each column
    basis : 1D_array_like
        List of strings that represents which basis transformation to perform on the datapoints

    Returns
    -------
    points : 2D_array_like
        Numpy array similar to the input parameter 'points' with the required basis transformation added
    """
    for b in basis:
        if(b == 'x^2'):
            new_value = points[:, 0] ** 2
            points = np.column_stack((points, new_value))
        elif(b == 'y^2'):
            new_value = points[:, 1] ** 2
            points = np.column_stack((points, new_value))
        elif(b == 'x*y'):
            new_value = points[:, 0] * points[:, 1]
            points = np.column_stack((points, new_value))
        elif(b == 'Sin(x)'):
            new_value = np.sin(points[:, 0])
            points = np.column_stack((points, new_value))
        elif(b == 'Sin(y)'):
            new_value = np.sin(points[:, 1])
            points = np.column_stack((points, new_value))
        elif(b == 'exp(-x^2)'):
            new_value = np.exp(-points[:, 0] ** 2)
            points = np.column_stack((points, new_value))
        else:
            new_value = np.exp(-points[:, 1] ** 2)
            points = np.column_stack((points, new_value))
    return points

def main():
    """
    Main funtion that run the model
    """
    st.title('Task 1 : Customizable Neural Network with Streamlit')
    st.subheader('Select Dataset')

    ## Selecting dataset
    dataset_points = select_dataset()

    ## User input controls
    with st.sidebar:
        num_layers = st.number_input('Number of hidden layers', min_value = 1, value = 1, step = 1)
        learning_rate = st.selectbox('Learning Rate', LEARNING_RATE_OPTIONS, index = LEARNING_RATE_OPTIONS.index(0.03))
        epochs = st.number_input('Number of epochs', min_value = 1, value = 100, step = 1)
        num_classes = st.number_input('Number of output nodes', min_value = 1, value = 2, step = 1)
        mc_dropout = st.checkbox('MC Dropout')

    train_test_split = st.slider('Ratio of Testing to Training', min_value = 0.1, max_value = 0.9, value = 0.2, step = 0.1)
    basis_transformation = st.multiselect('Basis Transformation', BASIS_TRANSFOMATION)
    plot_points(dataset_points)
    ## Creating the model
    model = create_model(num_layers, learning_rate, num_classes, mc_dropout)

    ## PreProcessing Dataset
    points, labels = get_points_labels(dataset_points)
    transformed_points = transform_points(points, basis_transformation)
    train_points, test_points, train_labels, test_labels = get_training_testing(transformed_points, labels, train_test_split)

    ## Train the model
    history = train_model(model, train_points, train_labels, epochs, test_points, test_labels)
    st.subheader('Training Finished')

    ## Getting the training and testing results
    model_results(history)

    ## Plot contours of probabilities
    plot_contour(model, points, labels, basis_transformation, mc_dropout)

if __name__ == '__main__':
    main()
