import numpy as np
import itertools
import sys
import traceback

try:
	import tensorflow.keras as keras
except ImportError:
	import keras

from caloGraphNN_keras import *

def open_file():
	'''
	Opens all 16 npz files and concatenates them
	'''
	events = []
	labels = []
	norm_events = []
	pad_events = []
	pad_labels = []
	for i in range(1, 17):
		data = np.load(f'{i}.npz', allow_pickle=True)
		events.append(data['events'])
		labels.append(data['labels'])
		norm_events.append(data['norm_events'])
		pad_events.append(data['pad_events'])
		pad_labels.append(data['pad_labels'])
	events = np.array(list(itertools.chain(*events)))
	labels = np.array(list(itertools.chain(*labels)))
	norm_events = np.array(list(itertools.chain(*norm_events)))
	pad_events = np.array(list(itertools.chain(*pad_events)))
	pad_labels = np.array(list(itertools.chain(*pad_labels)))
	return events, labels, norm_events, pad_events, pad_labels

class GravNetModel(keras.Model):
	'''
	Keras Model with GravNet layer
	'''
	def __init__(self, n_neighbors=5, n_dimensions=3, n_filters=5, n_propagate=15):
		super(GravNetModel, self).__init__()
		momentum = 0.99

		self.gravnet = self.add_layer(GravNet, n_neighbors, n_dimensions, n_filters, n_propagate, Name='gravnet')
		self.batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm')
		self.output_dense_1 = self.add_layer(keras.layers.Dense, 32, activation='relu', name='output_1')
		self.output_dense_2 = self.add_layer(keras.layers.Dense, 1, activation='sigmoid', name='output_2')

	def call(self, inputs):
		x = self.gravnet(inputs)
		x = self.batchnorm(x)
		x = self.output_dense_1(x)
		output = self.output_dense_2(x)
		return output

	def add_layer(self, cls, *args, **kwargs):
		layer = cls(*args, **kwargs)
		self._layers.append(layer)
		return layer
