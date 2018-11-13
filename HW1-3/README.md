# Homework 1-3

## Purpose: 

* Can network fit random labels?: 希望透過此實驗來瞭接 deep model 是相當有 power 的

## Data 簡介

Mnist dataset: 參考 [Center Loss Visualization project](https://github.com/machineCYC/SideProjects/tree/master/01-CenterLossVisualization) 的說明

## Summary 總結

### Can network fit random labels?

隨機從 mnist training datasets 中抽出 5000 筆當訓練資料，將相對應的 label 隨機更換。batch size 128, epoch 500, learning rate 1e-4。從實驗中可以發現 model 是可以將擁有錯誤 label 的訓練資料完整學習下來。

由下圖的 loss 和 accuracy 可以知道 model 過度的擬和訓練資料，訓練 loss 接近 0， accuracy 接近 1。但在驗證資料上面表現的就相當差。

由此實驗可以清楚的知道 deep model 是擁有將所有訓練資料記下來的能力，但也因為這樣會過擬和資料。

![](image/Random_label_accuracy.png)

![](image/Random_label_loss.png)

### Number of parameters v.s. Generalization

![](image/Nbr_para_gen_loss.png)

![](image/Nbr_para_gen_accuracy.png)

### Flatness v.s. Generalization - part1

### Flatness v.s. Generalization - part2

# Reference

* [原始作業說明](https://docs.google.com/presentation/d/18swR-wgvVWwiOds1cUrBbouAfd3YBRUC6RLUMoiUrns/edit#slide=id.p3)