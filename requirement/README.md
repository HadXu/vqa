# 配置caffe环境

> 需要的环境为Python2

使用的caffe为[bottom-up-attention](https://github.com/peteanderson80/bottom-up-attention)

## 原始的配置环境是有问题的

- 将CUDNN放出来，且用openBlas进行编译

- 如果遇到```fatal error: hdf5.h: No such file or directory compilation terminated.``` 请参考 [编译错误：fatal error: hdf5.h: No such file or directory compilation terminated.](https://blog.csdn.net/qq_38451119/article/details/81383266)

- 如果遇到```nvcc fatal : Unsupported gpu architecture 'compute_20'```请参考[nvcc fatal : Unsupported gpu architecture 'compute_20'](https://blog.csdn.net/fanhenghui/article/details/80092131)

- 如果遇到```undefined reference to cv::imread(cv::String const&, int)```
请将```Makefile.config```中的opencv注释取消掉

## 接下来

```
make -j8 && make pycaffe
```