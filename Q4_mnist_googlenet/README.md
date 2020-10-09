# NaiveGoogLeNet and GoogLeNet model training and test outputs analysis report

```
python train_googlenet.py
python train_naive_googlenet.py
```

The training process on the GoogLeNet with MNIST Datasets, takes around 1min30secs to train the model per epoch (based on the timer I added in the training code).
After 2 epochs training, the test accuracy reaches to 99%.

![Training Results of GoogLeNet](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-2/blob/dev/Q4_mnist_googlenet/Test%20Results/train_googlenet_result.png)

The training process on the NaiveGoogleNet with MNIST Datasets, takes around 2min20secs to train the model per epoch (based on the timer I added in the training code).
After 2 epochs training, the test accuracy reaches to 99%.

![Training Results of NaiveGoogLeNet](https://github.com/kuangzijian/UAlberta-Multimedia-Master-Program-MM811-2020-Assignment-2/blob/dev/Q4_mnist_googlenet/Test%20Results/train_naive_googlenet_result.png)


Comparing the training performance, the googleNet with naive inception is computationally expensive, and the training effiency is also lower than the GoogLeNet with v1 inception.

