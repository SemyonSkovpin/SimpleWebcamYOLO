import ultralytics.YOLO as YOLO
import cv2
import os
import tomlib



'''
The "model_to_use.toml" file

Currently used model is specified in "model_to_use.toml". 
Its variable "model_filename" is for the filename of the model in ultralytics library. Our program will look for 
its .pt weights in "models" folder and set it into "model" variable. Ultralytics library will automaticaly install it 
under that path if its not there.

Alternatively, in the "path_to_model" variable, you can directly set the path to a .pt weights you installed. It will take 
priority over the other method. Set it to '' when not using. 

- The "model_filename" needs to be one of ultralytics supported model filenames ('yolo26n.pt', 'yolo11x.pt' etc').

- The "path_to_model" needs to be '' or abslolute or relative path correct for your system. 
'''

# Get the current model
with open('model_to_use.toml', 'rb') as f:
	contents = tomlib.read(f)
	if contents['path_to_model'] != '':
		path_to_weights = contents['path_to_model']
	else:
		path_to_weights = os.path.join('models', contents['model_filename'])
model = YOLO(path_to_weights)



'''
The "labels_to_detect.txt"

Labels that you choose to be detected are written in "labels_to_detect.txt", separated by newlines. They will be filtered for 
those that are supported by the current model, before getting into the "chosen_labels" set.

User can write the labels directly to the file.
'''

# Get 2 sets: set of all supported labels, and its subset, which of them are checkmarked to be detected.
all_supported_labels = set(model.names.values())

chosen_labels = set()
with open("labels_to_detect.txt", 'r') as f:
	for label in f.read().split('\n'):
		if label in all_supported_labels:
			chosen_labels.add(label)














