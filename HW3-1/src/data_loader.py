import os
import cv2
import numpy as np

class GenerateDataSet():

    def __init__(self, image_dir, img_width, img_height, img_channel):
        self.img_width = img_width
        self.img_height = img_height
        self.img_channel = img_channel
        self.array_images = []

        for imgname in os.listdir(image_dir):
            img_path = os.path.join(image_dir, imgname)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (self.img_width, self.img_height), cv2.COLOR_BGR2RGB)
            self.array_images.append(img)

        self.image_nbr = len(self.array_images)
        self.index_in_epoch = 0
        self.N_epoch = 0

    def next_batch(self, batch_size):
        real_images = []
        for _ in range(batch_size):

            if self.index_in_epoch > self.image_nbr:
                random_idx = np.arange(0, self.image_nbr)
                np.random.shuffle(random_idx)

                self.array_images = self.array_images[random_idx]

                self.index_in_epoch = 0
                self.N_epoch += 1

            real_images.append(self.array_images[self.index_in_epoch])
            self.index_in_epoch += 1
        return real_images