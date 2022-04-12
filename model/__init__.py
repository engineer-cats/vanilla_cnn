import tensorflow as tf
import tensorflow.keras.layers as KL
def model():
	# MNIST dataset is 28 * 28
	input1 = tf.keras.layers.Input(shape = (28, 28, 1), name = "input1")
	conv1 = KL.Conv2D(filters = 6, kernel_size = (5, 5), activation = "tanh", padding = "same", name = "conv1")(input1)
	pool1 = KL.MaxPool2D(pool_size = (2, 2), name = "pool1")(conv1)
	conv2 = KL.Conv2D(filters = 16, kernel_size = (5, 5), activation = "tanh", padding = "same", name = "conv2")(pool1)
	pool2 = KL.MaxPool2D(pool_size = (2, 2), name = "pool2")(conv2)
	flatten1 = KL.Flatten()(pool2)
	dense1 = KL.Dense(400, activation = "tanh", name = "dense1")(flatten1)
	dense2 = KL.Dense(120, activation = "tanh", name = "dense2")(dense1)
	dense3 = KL.Dense(84, activation = "tanh", name = "dense3")(dense2)
	dense4 = KL.Dense(10, activation = "tanh", name = "dense4")(dense3)
	softmax1 = KL.Softmax()(dense4)
	conv_models = tf.keras.Model(input1, softmax1)
	return conv_models
