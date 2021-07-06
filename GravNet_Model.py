from toy_model import *
from GNN import *
from sklearn.metrics import roc_curve, auc

model = GravNetModel()
optimizer = keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

norm_events, labels = open_pkl_file()
padded_events = pad_events(norm_events)
padded_labels = pad_labels(labels)
frac_noise = [label.sum()/len(label) for label in padded_labels]
avg_frac_noise = np.array(sum(frac_noise)/len(padded_labels))

split = int(len(padded_events)/10)
train_events = padded_events[2*split:]
train_labels = padded_labels[2*split:]
val_events = padded_events[:split]
val_labels = padded_labels[:split]
test_events = padded_events[split:2*split]
test_labels = padded_labels[split:2*split]

hist = model.fit(x=train_events, y=train_labels, verbose=2, epochs=50, validation_data=(val_events, val_labels), shuffle=False, use_multiprocessing=True, workers=12, max_queue_size=16)

#test_events, test_labels, test_norm_events, test_padded_events, test_padded_labels = open_file(name='test')
#test_padded_events = pad_events(test_padded_events)
#test_padded_labels = pad_labels(test_padded_labels)
#frac_noise = [label.sum()/len(label) for label in test_padded_labels]
#avg_frac_noise = np.array(sum(frac_noise)/len(test_padded_labels))

pred_labels = model.predict(test_events).ravel()
fpr, tpr, thresholds = roc_curve(test_labels.ravel(), pred_labels)
auc = np.array(auc(fpr, tpr))

np.savez('roc', fpr=fpr, tpr=tpr, thresholds=thresholds, auc=auc, avg_frac_noise=avg_frac_noise)

