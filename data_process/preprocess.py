import numpy as np
import collections
import random
from keras.utils import to_categorical

buffer = np.load('../data/numpy_dict.npy', allow_pickle=True)[0]

key_order = []
for i in buffer:
    key_order.append(i)

Photo = collections.namedtuple('Photo', ['label', 'photo'])
photo_array = []

for i in range(len(key_order)):
    key = key_order[i]
    for j in range(100):
        photo_array.append(Photo(i, buffer[key][j]))

random.shuffle(photo_array)

labels = []
photos = []

for i in photo_array:
    labels.append(i.label)
    photos.append(i.photo)

npy_photos = np.stack(photos)

test_dict = {'labels': labels, 'photos': npy_photos}

np.save('../data/test_set.npy', np.array([test_dict]))

