import os
import json
import numpy as np

from src.constants import *


class DataSets():
    def __init__(self, image_feature_dir_path, captions):
        self._caption_map_feature = []
        self._caption = [] # 24232, 48
        self._feature = [] # 1450, 80, 4096
        self._video_id = [] # 1450
        self._maxlen_of_sentence = 0

        for idx, caption in enumerate(captions):
            video_id = caption["id"]
            self._video_id.append(video_id)
            video_captions = caption["caption"]

            image_feature = np.load(os.path.join(image_feature_dir_path, video_id + ".npy"))
            self._feature.append(image_feature)

            for sentence in video_captions:
                self._caption_map_feature.append(idx)
                self._caption.append(sentence)
                if len(sentence) > self._maxlen_of_sentence:
                    self._maxlen_of_sentence = len(sentence)
        
        self._caption_map_feature = np.asarray(self._caption_map_feature)
        self._video_id = np.asarray(self._video_id)
        self._caption = np.asarray(self._caption)
        self._feature = np.asarray(self._feature)
        self._nbr_of_caption = len(self._caption)
        self._nbr_of_feature = len(self._feature)
        self._dim_video_frame = len(self._feature[0])
        self._dim_video_feature = len(self._feature[0][0])
        self._train_index_th_epoch = 0

        self.shuffle_data()
    
    def next_batch(self, batch_size):
        image_feature = []
        image_caption = []
        image_caption_len = []

        for _ in range(batch_size):
            if self._train_index_th_epoch >= self._nbr_of_caption:
                self._train_index_th_epoch = 0
                self.shuffle_data()

            image_feature.append(self._feature[self._caption_map_feature[self._train_index_th_epoch]])
            image_caption.append(self._caption[self._train_index_th_epoch])
            image_caption_len.append(len(self._caption[self._train_index_th_epoch])-1)
            self._train_index_th_epoch += 1
        
        return np.asarray(image_feature), np.asarray(image_caption), np.asarray(image_caption_len)
    
    def next_inference_batch(self, batch_size):
        order = np.random.choice(self._nbr_of_feature, batch_size, replace=False)
        return np.asarray(self._feature[order]), np.asarray(self._video_id[order])

    def shuffle_data(self):
        order = np.arange(self._nbr_of_caption)
        np.random.shuffle(order)

        self._caption = self._caption[order]
        self._caption_map_feature = self._caption_map_feature[order]


def read_word2index(dic_full_path):
    dictionary = {}
    with open(dic_full_path, "r") as f:
        for data in f:
            [index, word, frequence] = data.split(" ")
            dictionary[word] = int(index)
    return dictionary


def read_index2word(dic_full_path):
    dictionary = {}
    with open(dic_full_path, "r") as f:
        for data in f:
            [index, word, frequence] = data.split(" ")
            dictionary[int(index)] = word
    return dictionary


def read_caption(caption_full_path):
    with open(caption_full_path, "r") as f:
        data = json.load(f)
    return data

def parse_index_sentence(caption_index, index2word):
    word_sentence = []
    for index in caption_index:
        word = index2word[index[0]]
        if word not in [EOS_WORD, BOS_WORD, UNK_WORD]:
            if word == "":
                word = "."
            word_sentence.append(word)
    return word_sentence

