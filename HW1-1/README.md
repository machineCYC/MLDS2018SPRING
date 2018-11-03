# Homework 1-1

## Purpose:

* Simulate a function: 本次作業希望透過模擬一個函數來了解 shallow structure 和 deep structure 之間的差異。試著去了解 deep structure 相對於 shallow 來的有 power。

* Train  on actual task:

## Data 簡介

* Simulate a function: 透過 numpy 的函數 sinc 生成 <a href="https://www.codecogs.com/eqnedit.php?latex=\frac{\sin&space;5\pi&space;x&space;}{5\pi&space;x&space;}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\frac{\sin&space;5\pi&space;x&space;}{5\pi&space;x&space;}" title="\frac{\sin 5\pi x }{5\pi x }" /></a> 共 10000 筆。x 軸範圍在 0.0001 ~ 1 之間，y 軸在 -0.2 ~ 1。如下圖所示:

![](image/target_model.png) 

* Train on actual task: 參考 [Center Loss Visualization](https://github.com/machineCYC/SideProjects/tree/master/01-CenterLossVisualization) project 的說明

## Summary 總結

### Simulate a function

我們嘗試了三個不同深度的模型，每一層均是 fully connection。在參數量接近的情況下，了解越深的模型可以模擬越複雜的函數，或者是說可以將函數配式的越好。模型結構如下:

| model | # of layers | # of parameters |
| --- | --- | --- |
| shallow model | 3 | 571 |
| medium model | 6 | 572 |
| deep model | 9 | 571 |

下圖為三個模型 training 過程中 loss 的狀況。可以觀察到 deep 模型所模擬出來的函數跟目標函數間的誤差最小。

![](image/loss.png)


下圖為三個模型 predict 的狀況。可以觀察到 deep 模型幾乎跟目標函數相同，而 medium 模型在波峰或波谷的地方預測的狀況比較差。shallow 只能捕捉到大方向的趨勢，對於細節的部分則完全不行。

![](image/predict.png)


### Train on actual task

我們嘗試了三個不同深度的模型，每一層均是 convolution + relu，最後一層是 fully connection + softmax。在參數量接近的情況下，了解不同深度的模型在處理相同問題時的差異。模型結構如下:

| model | # of layers | # of parameters |
| --- | --- | --- |
| cnn shallow model | 3 | 101560 |
| cnn medium model | 5 | 101266 |
| cnn deep model | 7 | 101398 |

下圖為三個模型 training 過程中 loss 的狀況。三個模型最後的 loss 都相當接近 0，但可以觀察到 shallow model 的 loss 還是稍微高一些。另外一點是 shallow model loss 下降的速度相對於 deep model 和 medium model 明顯來的慢。

![](image/cnn_loss.png)

下圖為三個模型 training 過程中 accuracy 的狀況。模型最後的 accuracy 其實都差不多，而從 accuracy 提升的速度可以知道越深的模型順度會越快，但越深的 model 也因為參數越多，在第一個 epoch 表現也比較不理想。

![](image/cnn_accuracy.png)

在 Simulate a function 的任務中，可以清楚的知道越深的模型可以處理越複雜的函數。Actual task 中得知 deep model 在 loss 下降和 accuracy 上升速度是比較快的，另外 mnist 這個 dataset 是一個相對簡單的 dataset，所以越深的模型優勢不是那麼的明顯。

# Reference

* [原始作業說明](https://docs.google.com/presentation/d/1VllCXAZ_DCN409MpmbsDV73wopoUt4gxyo47kPC1NGQ/edit#slide=id.p3)
