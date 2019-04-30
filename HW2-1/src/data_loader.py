import os
import numpy as np


class DataSets():
    def __init__(self, image_feature_dir_path, captions, BOS_VALUE, EOS_VALUE):
        self._caption_map_feature = []
        self._caption = []
        self._feature = []
        self._maxlen_of_sentence = 0

        for idx, caption in enumerate(captions):
            video_id = caption["id"]
            video_captions = caption["caption"]

            image_feature = np.load(os.path.join(image_feature_dir_path, video_id + ".npy"))
            self._feature.append(image_feature)

            for sentence in video_captions:
                sentence += [EOS_VALUE]
                sentence = [BOS_VALUE] + sentence
                self._caption_map_feature.append(idx)
                self._caption.append(sentence)
                if len(sentence) > self._maxlen_of_sentence:
                    self._maxlen_of_sentence = len(sentence)
        
        self._caption = np.asarray(self._caption)
        self._feature = np.asarray(self._feature)
        self._nbr_of_caption = len(self._caption)
        self._index_th_epoch = 0

        self.shuffle_data()
    
    def next_batch(self, batch_size):
        image_feature = []
        image_caption = []

        for _ in range(batch_size):
            if self._index_th_epoch >= self._nbr_of_caption:
                self._index_th_epoch = 0
                self.shuffle_data()

            image_feature.append(self._feature[self._caption_map_feature[self._index_th_epoch]])
            image_caption.append(self._caption[self._index_th_epoch])
            self._index_th_epoch += 1
        
        return np.asarray(image_feature), np.asarray(image_caption)
    
    def shuffle_data(self):
        order = np.arange(self._nbr_of_caption)
        np.random.shuffle(order)

        self._caption = self._caption[order]
        self._caption_map_feature = self._caption_map_feature[order]

