# demonstration of object localization
# digits classification and bound box with MNIST dataset

# Author: Wenxu Zhang
# created: Apr 5, 2022
# last motified: Apr 5, 2022

import tensorflow as tf
import keras
from preprocess import normalization
import matplotlib.pyplot as plt
from model import model
import numpy as np
from preprocess import one_hot_coding
import os
from tensorflow.keras import backend as K
import tensorflow as tf
from sklearn.decomposition import PCA

def normalize_train_test_data(train_data, test_data):
	normalized_train_data = normalization(train_data)
	normalized_test_data = normalization(test_data)
	return normalized_train_data, normalized_test_data

def one_hot_labels(train_label, test_label):
	one_hot_train_label = one_hot_coding(train_label)
	one_hot_test_label = one_hot_coding(test_label)
	return one_hot_train_label, one_hot_test_label

def training(x_train, y_train, x_test, y_test):
	HOMEDIRECTORY = os.getcwd()

	plt.imshow(x_test[0, :, :])
	plt.show()
	print(y_test[0, :])
	conv_model = model()
	print(conv_model.summary())	

	#step = tf.Variable(0, trainable=False)
	#boundaries = [1, 2, 3, 4]
	#values = [0.01, 0.001, 0.0001, 0.00001, 0.000001]
	#lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)

	conv_model.compile(optimizer = tf.keras.optimizers.RMSprop(), loss = tf.keras.losses.CategoricalCrossentropy(), metrics = [tf.keras.metrics.CategoricalAccuracy()])
	conv_model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size = 128,  epochs = 10)
	conv_model.save(os.path.join(HOMEDIRECTORY, 'digit_classification.h5'))


def data_prep(is_onehot):
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
	(x_train, x_test) = normalize_train_test_data(x_train, x_test)

	x_train = x_train[:, :, :, np.newaxis]
	x_test = x_test[:, :, :, np.newaxis]
	if is_onehot:
		(y_train, y_test) = one_hot_labels(y_train, y_test)

	return x_train, y_train, x_test, y_test

def digits_pca(x_train, y_train):
	HOMEDIRECTORY = os.getcwd()
	conv_model = tf.keras.models.load_model(os.path.join(HOMEDIRECTORY, 'digit_classification.h5'))
	get_fc_layer_output = K.function([conv_model.input], [conv_model.get_layer('dense4').output])
	fc_output = get_fc_layer_output(x_train)[0] # first element is value and second element is dtype
	# print(fc_output)
	pca = PCA(n_components=2)
	principalComponents = pca.fit_transform(fc_output)
	fig, ax = plt.subplots()
	ax.scatter(principalComponents[:, 0], principalComponents[:, 1], c = y_train, s = 1)
	ax.legend()
	plt.show()

def main():
	#x_train, y_train, x_test, y_test = data_prep(is_onehot = True)
	#training(x_train, y_train, x_test, y_test)
	x_train, y_train, x_test, y_test = data_prep(is_onehot = False)
	digits_pca(x_train, y_train)


if __name__ == "__main__":
	print("hello world!")
	main()




