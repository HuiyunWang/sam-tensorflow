# sam-tensorflow

## Paper
Huiyun Wang, Youjiang Xu, [Yahong Han](http://cs.tju.edu.cn/faculty/hanyahong/). "Spoting and Aggregating Salient Regions for Video Captioning." 
Full Paper in ACM MM,2018.

## Reference
If you find this useful in your work, please consider citing the following reference:
```c
@inproceedings{samwangMM18,
    title = {Spoting and Aggregating Salient Regions for Video Captioning},
    author = {Wang, Huiyun and Xu, Youjiang and Han, Yahong},
    booktitle = {Proceedings of the ACM International Conference on Multimedia (ACM MM)},
    year = {2018}
}
```



## Framework
![](https://github.com/HuiyunWang/sam-tensorflow/blob/master/figure/framework.png)

The framework of our method. We firstly extract feature maps from the convolutional layer of CNN. Then, in the ‘Spot Module’, we automatically learn the saliency value of each location to separate salient regions from video content as the foreground and the
rest as background by two operations of ‘hard separation’ and ‘soft separation’, respectively. To aggregate the foreground/background
descriptors into a discriminative spatio-temporal representation, in the ‘Aggregate Module’, we devise a trainable video VLAD process
to learn the aggregation parameters. Finally, we utilize the attention mechanism to decode the spatio-temporal representations of
diﬀerent regions into video description.

## Datasets
The datasets used in the paper are available at the following links:

* Microsoft Video Description Corpus (MSVD):
[Raw Data Page](http://www.cs.utexas.edu/users/ml/clamp/videoDescription/), [Raw Corpus](https://www.microsoft.com/en-us/download/details.aspx?id=52422&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F38cf15fd-b8df-477e-a4e4-a4680caa75af%2Fdefault.aspx), [Processed Data](https://www.dropbox.com/sh/4ecwl7zdha60xqo/AAC_TAsR7SkEYhkSdAFKcBlMa?dl=0).

* Montreal Video Annotation Description (M-VAD) Dataset:
[Data Page](http://www.mila.umontreal.ca/Home/public-datasets/montreal-video-annotation-dataset).

## Training
The code is implemented on tensorflow 1.0.1.

The scrips for training and testing of our method can be seen [run.sh](https://github.com/HuiyunWang/sam-tensorflow/blob/master/run.sh).


## Examples
![](https://github.com/HuiyunWang/sam-tensorflow/blob/master/figure/visualization.png)

Some examples of our method.
