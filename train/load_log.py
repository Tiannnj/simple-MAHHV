import os
import json
import tensorflow as tf


def get_loss_from_tfevent_file(tfevent_filename):
    """

    :param tfevent_filename: the name of one tfevent file
    :return: loss (list)
    """
    loss_val_list = []
    for event in tf.train.summary_iterator(tfevent_filename):
        for value in event.summary.value:
            # print(value.tag)
            if value.HasField('simple_value'):
                if value.tag == "loss":
                    # print(value.simple_value)
                    loss_val_list.append(value.simple_value)

    return loss_val_list



if __name__ == "__main__":
    get_loss_from_tfevent_file()