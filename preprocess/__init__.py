# image preprocess

# Author: Wenxu Zhang
# created: Apr 7, 2022
# last motified: Apr 7, 2022



import numpy as np

def one_hot_coding(ids):
	one_hot_ids = np.zeros((ids.size, ids.max() + 1))
	one_hot_ids[np.arange(ids.size), ids] = 1
	return one_hot_ids

def normalization(images, axis = (1, 2)):

	"""
	For batch of images with dimension N * h * w, we normalize all N frames.
	For every frames, subtract mean value first and then divide standard deviation.

	Input: images (N * h * w), axis: normalized axis
	Output: normalized_images (N * h * w)
	"""

	if axis == (1, 2):
		normalized_images = images - images.mean(axis = axis)[:, np.newaxis, np.newaxis]
		normalized_images = normalized_images / np.std(normalized_images, axis = axis)[:, np.newaxis, np.newaxis]
	else:
		print("WARNING: add functions to deal with axis other than (1, 2)")

	return normalized_images

def image_add_last_dimension():
	
	"""
	train dataset is 10k (N) * 28 * 28
	after adding the dimension, it becomes 10k * 28 * 28 * 1

	Input: images N * h * w
	Ouput: images N * h * w * 1
	"""
