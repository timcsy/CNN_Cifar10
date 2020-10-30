# Recognizing Cifar10 using VGG16 / All-CNN

## Requirements
- Python 3 64-bit
- Dependency
	```
	tkinter tensorflow matplotlib numpy
	```

## Usage
run:
```
python cnn.py
```

## Result
![](images/result.png)

## 心得
- Colab 可以用 GPU 加速！！
- 用 VGG16 訓練一開始，Accuracy 結果都趨近於0.1（隨便亂猜的意思）
- VGG16 因為會 Gradient vanish，所以要做 batch normalize
- 其實也有其他的 Model 可以實作，我也有用 All-CNN 來寫

## Reference
- [Convolutional Neural Network (CNN)](https://www.tensorflow.org/tutorials/images/cnn)
- [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
  - Not used
  - Found the following reference
- [STRIVING FOR SIMPLICITY: THE ALL CONVOLUTIONAL NET](https://arxiv.org/pdf/1412.6806.pdf)
  - Use the All-CNN-C Model
  - [All-CNN](https://github.com/PAN001/All-CNN)
- [python - 使用子类模型时，model.summary()无法打印输出形状](https://www.coder.work/article/1258695)