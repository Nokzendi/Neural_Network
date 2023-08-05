import math
from sklearn.datasets import make_moons
import numpy as np

np.random.seed(42)

class DataPoint:
    """
    Class to store the datapoints with the output class labels

    Parameters
    ----------
    x : float
        x coordinate of the data point
    y : float
        y coordinate of the data point
    label : int
        output class label
    """
    def __init__(self, x, y, label):
        self.x = x
        self.y = y
        self.label = label

def rand_uniform(min_val, max_val):
    """
    Numpy's uniform random function
    """
    return np.random.uniform(min_val, max_val)

def normal_random(mean, variance):
    """
    Numpy's normal random function
    """
    return mean + np.random.normal(loc=0.0, scale=variance)

def classify_circle_data(num_samples, noise):
    """
    Function to create a dataset of two concentric circles. The datapoints in the inner circle and outer circle gets
    different labels

    Parameters
    ----------
    num_samples : int
        number of datapoints to be included in the dataset
    noise : float
        amount of noise to use while creating the dataset
    """
    points = []
    radius = 5

    def get_circle_label(p, center):
        """
        Function to label the datapoints based on the distance from the center. Divided into equal halves.

        Parameters
        ----------
        p : DataPoint object
            datapoint to label
        center : DataPoint object
            center of the circle
        """
        return 1 if math.sqrt((p.x - center.x) ** 2 + (p.y - center.y) ** 2) < (radius * 0.5) else 0
    
    # Generate positive points inside the circle.
    for i in range(num_samples // 2):
        r = np.random.uniform(0, radius * 0.5)
        angle = np.random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label(DataPoint(x + noise_x, y + noise_y, None), DataPoint(0, 0, None))
        points.append(DataPoint(x, y, label))

    # Generate negative points outside the circle.
    for i in range(num_samples // 2):
        r = np.random.uniform(radius * 0.7, radius)
        angle = np.random.uniform(0, 2 * math.pi)
        x = r * math.sin(angle)
        y = r * math.cos(angle)
        noise_x = np.random.uniform(-radius, radius) * noise
        noise_y = np.random.uniform(-radius, radius) * noise
        label = get_circle_label(DataPoint(x + noise_x, y + noise_y, None), DataPoint(0, 0, None))
        points.append(DataPoint(x, y, label))

    return points

def classify_spiral_data(num_samples, noise):
    """
    Function to create a spiral dataset

    Parameters
    ----------
    num_samples : int
        number of datapoints to be included in the dataset
    noise : float
        amount of noise to use while creating the dataset
    """
    points = []
    n = num_samples // 2

    def gen_spiral_point(i, n, delta_t, label, noise):
        r = i / n * 5
        t = 1.75 * i / n * 2 * np.pi + delta_t
        x = r * np.sin(t) + rand_uniform(-1, 1) * noise
        y = r * np.cos(t) + rand_uniform(-1, 1) * noise
        return DataPoint(x, y, label)

    for i in range(n):
        points.append(gen_spiral_point(i, n, delta_t=0, label=1, noise=noise))

    for i in range(n):
        points.append(gen_spiral_point(i, n, delta_t=np.pi, label=0, noise=noise))

    return points

def classify_xor_data(num_samples, noise):
    """
    Function to create a XOR dataset

    Parameters
    ----------
    num_samples : int
        number of datapoints to be included in the dataset
    noise : float
        amount of noise to use while creating the dataset
    """
    points = []
    def get_xor_label(p):
        return 1 if p.x * p.y >= 0 else 0
    for i in range(num_samples):
        x = rand_uniform(-5, 5)
        padding = 0.3
        x += padding if x > 0 else -padding
        y = rand_uniform(-5, 5)
        y += padding if y > 0 else -padding
        noise_x = rand_uniform(-5, 5) * noise
        noise_y = rand_uniform(-5, 5) * noise
        label = get_xor_label(DataPoint(x + noise_x, y + noise_y, None))
        points.append(DataPoint(x, y, label))
    return points

def classify_two_gauss_data(num_samples, noise):
    """
    Function to create a two gauss dataset

    Parameters
    ----------
    num_samples : int
        number of datapoints to be included in the dataset
    noise : float
        amount of noise to use while creating the dataset
    """
    points = []

    variance_scale = lambda x: 0.5 + 3.5 * (x - 0) / (0.5 - 0)
    variance = variance_scale(noise)

    def gen_gauss(cx, cy, label):
        for i in range(num_samples // 2):
            x = normal_random(cx, variance)
            y = normal_random(cy, variance)
            points.append(DataPoint(x, y, label))

    gen_gauss(2, 2, 1)
    gen_gauss(-2, -2, 0)

    return points

# Create the moons dataset
def classify_moons_data(num_samples, noise):
    """
    Function to create a moons dataset

    Parameters
    ----------
    num_samples : int
        number of datapoints to be included in the dataset
    noise : float
        amount of noise to use while creating the dataset
    """
    points = []
    X, y = make_moons(n_samples=num_samples, noise=noise, random_state=42)
    for i in range(len(y)):
        points.append(DataPoint(X[i][0], X[i][1], y[i]))
    return points

