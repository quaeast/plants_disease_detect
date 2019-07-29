import numpy as np


def read_img_package():
    buffer = np.load('../numpy_dict.npy', allow_pickle=True)
    print(type(buffer[0]))
    for i in buffer[0]:
        print('>-----------------------')
        print(i)
        print(buffer[0][i].shape)

