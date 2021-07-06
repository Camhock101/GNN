import numpy as np
import itertools
import sys
import traceback
import pickle

try:
	import tensorflow.keras as keras
except ImportError:
	import keras

from caloGraphNN_keras import *
from spektral.layers.convolutional import edge_conv

def open_npz_file(name=None):
	'''
	Opens all 16 npz files and concatenates them
	'''
	events = []
	labels = []
	norm_events = []
	pad_events = []
	pad_labels = []
	for i in range(1, 17):
		data = np.load(f'{name}_{i}.npz', allow_pickle=True)
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

def open_pkl_file():
	'''
	Opens all 16 pkl files, normalizes the event features,
	and concatenates them to be fed to the GNN.
	'''

	events = []
	labels = []
	for i in range(1, 17):
		file = open(f'{i}.pkl', 'rb')
		data = pickle.load(file)
		file.close()
		for j in range(len(data)):
			hit_info = data[j]['hits']
			labels.append(hit_info[:,-1])
			events.append(hit_info[:, :5])
	means = np.mean(list(itertools.chain(*events)), axis=0)
	stds = np.std(list(itertools.chain(*events)), axis=0)
	norm_events = [(e - means)/stds for e in events]
	return norm_events, labels
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

class EdgeConvModel(keras.Model):
	'''
	A model using Spektral's EdgeConv layer
	'''
	def __init__(self, channels=32):

		super(EdgeConvModel, self).__init__()
		momentum = 0.99

		self.edgeconv = self.add_layer(edge_conv.EdgeConv, channels, name='edge_conv')
		self.batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm')
		self.output_dense_1 = self.add_layer(keras.layers.Dense, 16, activation='relu', name='output_1')
		self.output_dense_2 = self.add_layer(keras.layers.Dense, 1, activation='sigmoid', name='output_2')

	def call(self, inputs):
		print(inputs)
		if len(inputs) == 2:
#			x, a = inputs
			x = inputs[0]
			a = inputs[1]
		else:
#			x, a, _ = inputs
			x = inputs[1]
			a = inputs[2]
#		x, a = inputs
		print(x)
		print(a)
		x = self.edgeconv([x, a])
		x = self.batchnorm(x)
		x = self.output_dense_1(x)
		output = self.output_dense_2(x)
		return output

	def add_layer(self, cls, *args, **kwargs):
		layer = cls(*args, **kwargs)
		self.layers.append(layer)
		return layer


class GarNetModel(keras.Model):
    '''
	Keras Model with GarNet layer
	'''
    def __init__(self, aggregators=4, filters=20, propagate=22, input_format='x'):
        super(GarNetModel, self).__init__()
        momentum = 0.99

        self.garnet = self.add_layer(GarNet, aggregators, filters, propagate, input_format='x', mean_by_nvert=True, name='garnet')
        self.batchnorm = self.add_layer(keras.layers.BatchNormalization, momentum=momentum, name='batchnorm')
        self.output_dense_1 = self.add_layer(keras.layers.Dense, 32, activation='relu', name='output_1')
        self.output_dense_2 = self.add_layer(keras.layers.Dense, 1, activation='sigmoid', name='output_2')
        
    def call(self, inputs):
        x = self.garnet(inputs)
        x = self.batchnorm(x)
        x = self.output_dense_1(x)
        output = self.output_dense_2(x)
        return output
    
    def add_layer(self, cls, *args, **kwargs):
        layer = cls(*args, **kwargs)
        self._layers.append(layer)
        return layer
