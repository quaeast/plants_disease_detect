import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

# img = mpimg.imread('./Plant2/Pepper__bell___Bacterial_spot/*')  # 读取和代码处于同一目录下的 lena.png




def package_img(dir_name):
    img_list_string = os.listdir('./Plant2/'+dir_name)
    img_matrix_list = []
    for i in img_list_string:
        img_buffer = mpimg.imread('./Plant2/'+dir_name+'/'+i)
        if img_buffer.shape != (256, 256, 3):
            continue
        img_matrix_list.append(img_buffer)
    np_stack = np.stack(img_matrix_list, 0)
    return np_stack

def show_img(img):
    plt.imshow(img)  # 显示图片
    plt.axis('off')  # 不显示坐标轴
    plt.show()


def package_dirs():
    ls = os.listdir('./Plant2')
    img_dict = {}
    for i in ls:
        print(i)
        img_arr_buffer = package_img(i)
        img_dict[i] = img_arr_buffer
    np.save('numpy_dict.npy', np.array([img_dict]))


# package_dirs()


# read_img_package()
