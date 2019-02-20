本文是将sphereface论文中训练好的sphere20模型通过webcaricature数据集进行fine-tuning。


## 运行命令
> 数据集转正：python warp.py
> 微调模型：python transfer_learning.py
> 测试：python wc_eval.py

## 3.数据预处理

### 3.1 WebCaricature数据
&emsp;&emsp;本实验采用的数据集为WebCaricature数据集，包含了252个人物的6042张漫画和5974张图片，如图1所示![](elements/datasets.png)


### 3.2 转正
&emsp;&emsp;本研究选择了SphereFace中的Landmark对数据集中的人脸进行转正，其对应坐标如表1所示，需要注意WebCaricature数据集中只有眼睛左右角的坐标，这里对眼睛左右角的坐标取均值作为眼睛的坐标。
<center>表1 5个landmark坐标信息示例</center>

|landmark名称|landmark坐标|
|---|---|
|左眼|30.2946, 51.6963|
|右眼|65.5318, 51.5014|
|鼻尖|48.0252, 71.7366|
|左边嘴巴|33.5493, 92.3655|
|右边嘴巴|62.7299, 92.2041|

&emsp;&emsp;接着我们根据sphereface中的landmark和数据集中的landmarks得到仿射矩阵，用来对图片以及landmarks进行放射变换，与此同时在仿射变换后的图片中将人脸部分裁剪出来，便得到了最终处理好的图片。


## 4.网络模型
### 4.1 训练
&emsp;&emsp;本研究选择了SphereFace作为人脸识别的模型，利用预训练好的SphereFace权重，针对WebCaricature数据集进行微调后，来判断人脸识别的准确度。训练数据是WebCaricature数据集中非严格人脸验证规范中的训练集。训练集中图片在人脸转正之后，高度为200，宽度为232，训练时从转正后的图片中裁剪出高度为112，宽度为96的图片，进行随机的翻转，并把每个位置的像素值减去127.5，与128相除后，传入网络。训练前，把SphereFace中前部分的权重固定，重新初始化了最后一层全连接层（fc6），这样会有参数更新的几个模块分别是：conv4_1、relu4_1、conv4_2、relu4_2、conv4_3、relu4_3、fc5、fc6。<br>
&emsp;&emsp;在训练过程中使用的loss函数是sphereface中提出来的A_softmax loss 。


### 4.2 测试
&emsp;&emsp;在对sphereface模型进行微调之后，我们将在实验中进行性能测试。使用到的数据集是WebCaricature数据集中非严格人脸验证规范中的训练对。在本实验中使用交叉验证的方式进行，folds设置为10，最终的性能指标即为10次结果的平均值。
&emsp;&emsp;因为是非严格的数据集，但是我们需要的数据是人脸照片和漫画图片的数据对，此时的数据对既可以是相同人的数据对也可以是不同人的。我们将读取出来的数据对输入到训练好的网络中得到输出结果，根据输出结果来计算两张图片的cos距离。

## 5.实验细节
&emsp;&emsp;训练时，batch的大小为128，使用SGD参数更新策略 ，SGD的动量参数设置为0.9，学习率最初是0.01，分别在第16000、24000次迭代的时候，学习率衰减为先前的0.1，最后在第28000次迭代的时候结束训练。权重衰变（weight decay）值设置为0.0005。