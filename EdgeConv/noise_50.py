import os
import numpy as np

from spektral.data import Dataset, Graph

class Noise_50(Dataset):

	def __init__(self, data_path):

		self.data_path = data_path
		super().__init__()

	def download(self):

		data = np.load(self.data_path, allow_pickle=True)
		os.mkdir(self.path)
		norm_events = data['norm_events']
		labels = data['labels']
		num_graphs = len(norm_events)

		for i in range(num_graphs):
			x = norm_events[i]
			y = labels[i]
			num_nodes = norm_events[i].shape[0]
			a = np.ones((num_nodes, num_nodes))
			filename = os.path.join(self.path, f'graph_{i}')
			np.savez(filename, x=x, y=y, a=a)

	def read(self):
		output = []
		for i in range(20000):
			data = np.load(os.path.join(self.path, f'graph_{i}.npz'))
			output.append(Graph(x=data['x'], a=data['a'], y=data['y']))
		return output
