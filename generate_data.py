from toy_model import *
import sys

if len(sys.argv) != 3:
	print("USAGE: %s <output_filename> <# of events>" %sys.argv[0])
	sys.exit(1)

output_fname = sys.argv[1]
num_events = int(sys.argv[2])

model = toy_experiment()
events, labels, truth = model.generate_events(num_events)
norm_events = normalize(events)
pad_events = pad_events(norm_events)
pad_labels = pad_labels(labels)
np.savez(f'{output_fname}', pad_events=pad_events, pad_labels=pad_labels, events=events, labels=labels, norm_events=norm_events)
