from toy_model import *
from new_block_GNN import *
from sklearn.metrics import roc_curve, auc

model = GravNetModel()
optimizer = keras.optimizers.Adam(0.0001)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

norm_events, labels = open_pkl_file(name='train')
padded_events = pad_events(norm_events)
padded_labels = pad_labels(labels)
#frac_noise = [label.sum()/len(label) for label in padded_labels]
#avg_frac_noise = np.array(sum(frac_noise)/len(padded_labels))

split = int(len(padded_events)/10)
train_events = padded_events[split:]
train_labels = padded_labels[split:]
val_events = padded_events[:split]
val_labels = padded_labels[:split]
#test_events = padded_events[split:2*split]
#test_labels = padded_labels[split:2*split]

hist = model.fit(x=train_events, y=train_labels, verbose=2, epochs=50, validation_data=(val_events, val_labels), shuffle=False, use_multiprocessing=True, workers=12, max_queue_size=16)

test_norm_events, test_labels = open_pkl_file(name='test')
test_padded_events = pad_events(test_norm_events)
test_padded_labels = pad_labels(test_labels)
frac_noise = [label.sum()/len(label) for label in test_padded_labels]
avg_frac_noise = np.array(sum(frac_noise)/len(test_padded_labels))

pred_prob = model.predict(test_padded_events).ravel()

test_padded_energies = test_padded_events[:, :, -1].ravel()
test_padded_idx = np.nonzero(test_padded_energies)[0]
real_labels = np.concatenate(test_labels)
real_pred_prob = pred_prob[test_padded_idx]


fpr, tpr, thresholds = roc_curve(real_labels, real_pred_prob)
auc = np.array(auc(fpr, tpr))

idx_75 = np.argmin(abs(tpr - 0.75))
idx_90 = np.argmin(abs(tpr - 0.90))
idx_98 = np.argmin(abs(tpr - 0.98))

fpr_75 = fpr[idx_75]
fpr_90 = fpr[idx_90]
fpr_98 = fpr[idx_98]

thresh_75 = thresholds[idx_75]
thresh_90 = thresholds[idx_90]
thresh_98 = thresholds[idx_98]

np.savez('roc_new_block_GravNet', fpr=fpr, real_pred_prob=real_pred_prob, tpr=tpr, thresholds=thresholds, auc=auc, avg_frac_noise=avg_frac_noise, fprs=[fpr_75, fpr_90, fpr_98], thresh=[thresh_75, thresh_90, thresh_98])

