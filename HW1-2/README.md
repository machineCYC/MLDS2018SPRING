# Homework 1-2

## Purpose: 

本次的作業主要是透過 optimization 的過程中了解 loss、accuracy、gradients 的變化以及所找到的這個 critical point 是否為 local minimal。

* Visualize the Optimization Process: 希望藉由視覺化的方式來了解訓練過程中 loss surface 和 accuracy 之間的變化

* Observe Gradient Norm During Training: 觀察訓練過程中 gradient、loss 和 accuracy 的變化

* What Happened When Gradient is Almost Zero: 希望透過 gradients 接近 0 時的 hessian 來觀察這個點是否為 local minimal

## Data 簡介

Mnist dataset: 參考 [Center Loss Visualization project](https://github.com/machineCYC/SideProjects/tree/master/01-CenterLossVisualization) 的說明

## Summary 總結

### Visualize the Optimization Process

隨機挑選 mnist 2 千筆 data，來 trian 一個 dnn model，總共訓練 30 個 epochs。並在每 3 次 epochs 結束時蒐集所有的參數，再透過 pca 將參數投影至 2 dim 空間以視覺化的方式呈現參數與 accuracy 之間的關係。反覆訓練相同的模型共 8 次。

由於每次訓練參數的起始值都會不一樣，導致每次訓練完 accuracy 也會不一樣，大約都在 90% 左右，另外由下圖的結果可以發現，參數起始值雖然不一樣，但卻一致的往右邊靠近，由此可知右邊 loss 相對左邊而言比較低。

![](image/visulization_weights.png)

### Observe Gradient Norm During Training

訓練過程中紀錄模型參數 gradient norm、loss 和 accuracy 的變化。此實驗隨機抓取 mnist dataset 中隨機 2000 筆資料，訓練 30 epochs。

由下圖可以觀察到 loss 越來越小的過程中，gradient 越來越大，似乎跟直覺不太一樣。使用 gradient base 的方式來訓練模型，當 loss 越來越小，會認為越來越近進 local minimum ，gradients 應該也要越來越小。目前猜測是因為訓練次數不夠多，導致 loss 還不夠小，因此 gradient 還很大步伐的更新參數。

![](image/visulization_grads.png)

### What Happened When Gradient is Almost Zero

本次實驗透過一個小型的神經網絡來擬和一個函數，共訓練 50 個 epochs，在模型訓練結束時再對 gradient 做最佳化，當 gradient 很接近 0 時，再對參數計算 hessian matrix，並透過 eigen value 的方式來理解 train 出來的模型是收斂到 local minimum 還是 saddle point。

由下圖可以發現，當 loss 越來越小時，eigen value 大於 0 的比例越來越大，也就是越像 local minimum。比例越來越低則比較傾相 saddle point。

![](image/min_ratio.png)

# Reference

* [原始作業說明](https://docs.google.com/presentation/d/1siUFXARYRpNiMeSRwgFbt7mZVjkMPhR5od09w0Z8xaU/edit#slide=id.p3)
