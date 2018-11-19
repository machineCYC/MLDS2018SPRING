import os
import numpy as np
import tensorflow as tf


def count_model_parameters():
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
        print("Total parameters:{}".format(total_parameters))
    return total_parameters


def np_save_list(save_fullpath, save_list):
    save_array = np.asarray(save_list)
    if not os.path.exists(os.path.dirname(save_fullpath)):
        os.makedirs(os.path.dirname(save_fullpath))
    np.save(save_fullpath, save_array)
    print("Save path: ", save_fullpath)
