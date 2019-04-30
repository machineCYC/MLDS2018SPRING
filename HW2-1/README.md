# Homework 2-1

## Purpose:

* Video Caption Generation: 希望訓練一個模型可以讀取一段影片，並描述這段影片, 為了簡化問題

## Data 簡介

- Training: 1450 videos
- Testing: 100 videos

## Summary 總結

learning rate 0.01  太大導致 inference 時每個video都一樣的敘述

### Model description

### How to improve your performance


### Experimental results and settings

### Video Caption Generation

<div class="center">
    <img src="image/Random_label_accuracy.png" height="300px">
    <img src="image/Random_label_loss.png" height="300px">
</div>

![](image/Flatness_vs_gen_p1_inter.png)


# Reference

* [原始作業說明](https://docs.google.com/presentation/d/1AeHW6-VDchIbjBXrOPQpXek82L3bi5PR5RapbOhcw94/edit#slide=id.p3)

* [Neural Machine Translation (seq2seq) Tutorial(Tensorflow)](https://github.com/tensorflow/nmt)

* [Sequence to Sequence – Video to Text](http://www.cs.utexas.edu/users/ml/papers/venugopalan.iccv15.pdf)

* [BLEU](https://aclanthology.info/pdf/P/P02/P02-1040.pdf)

* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

* [RECURRENT NEURAL NETWORK REGULARIZATION](https://arxiv.org/pdf/1409.2329.pdf)

condition generation