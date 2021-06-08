from toy_model import *
import itertools
from GNN import *
from spektral.datasets.noise_50 import Noise_50
from spektral.data.loaders import DisjointLoader
from sklearn.metrics import roc_curve, auc

model = EdgeConvModel()
optimizer = keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

dataset = Noise_50('data/abald/GNN/noise_50/test_data.npz')
frac_noise = []
for i in range(len(dataset)):
	frac_noise.append(dataset[i]['y'].sum()/len(dataset[i]['y']))
avg_frac_noise = np.array(sum(frac_noise)/len(frac_noise))

split = int(len(dataset)/10)
train_loader = DisjointLoader(dataset[2*split:], batch_size=32, node_level=True)
val_loader = DisjointLoader(dataset[split:2*split], batch_size=32, node_level=True)
test_loader = DisjointLoader(dataset[:split], batch_size = 32, node_level=True)

test_labels = []
for i in range(split):
	test_labels.append(dataset[i]['y'])
test_labels = np.array(list(itertools.chain(*test_labels)))

print(train_loader.load())
hist = model.fit(train_loader.load(), steps_per_epoch=train_loader.steps_per_epoch, verbose=2, epochs=50, validation_data=val_loader.load(),
		 validation_steps=val_loader.steps_per_epoch, use_multiprocessing=True, workers=12, max_queue_size=16)

pred_labels = model.predict(test_loader.load(), steps_per_epoch=test_loader.steps_per_epoch).ravel()
fpr, tpr, thresholds = roc_curve(test_labels, pred_labels)
auc = np.array(auc(fpr, tpr))

np.savez('roc', fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc, avg_frac_noise=avg_frac_noise)

