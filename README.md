## 05-图像纹理识别

#### 计算机视觉之纹理

一幅图像的纹理是在图像计算中经过量化的图像特征。图像纹理描述图像或其中小块区域的空间颜色分布和光强分布。纹理特征的提取分为基于结构的方法和基于统计数据的方法。一个基于结构的纹理特征提取方法是将所要检测的纹理进行建模，在图像中搜索重复的模式。该方法对人工合成的纹理识别效果较好。但对于交通图像中的纹理识别，基于统计数据的方法效果更好。

CNN的卷积操作是以滑窗方式操作，充当局部特征提取器。输出的特征图保存了输入图片的相对的空间排列。再将得到的全局有序的特征级联送到FC层做分类， 这样的框架在图像分类，目标识别等获得巨大成功。但是这不适用于纹理识别，因为纹理识别需要描述特征分布的空间不变表示而不是级联。

1. 每个像素点形成一个纹理描述。

   找到纹理基元，纹理基元通常是由子元素构成的（点和条形），可以使用不同方向、尺度、相位的滤波寻找子元素，再通过找到的子元素的近邻来描述图像中的每个点（高斯滤波可实现）

2. 池化纹理表示

   如果直接使用基于局部的特征表示构建图像区域的特征，向量维数太大了；如果通过直方图表示区域特征的话，cell个数太多了，因为每个像素的纹理表示多少有些不同。

在纹理处理中一般有三个基本问题：

- 纹理分割把图片分成不同的部分，每部分内部具有相近的纹理问题
- 纹理合成寻找如何利用小范例图像构造大片纹理区域的方法
- 纹理恢复形状包括由图像纹理恢复表面的方向和表面的形状。

#### 格拉姆矩阵

来自知乎 90后后生：Gram Matrix实际上可看做是feature之间的偏心协方差矩阵（即没有减去均值的协方差矩阵），在feature map中，每一个数字都来自于一个特定滤波器在特定位置的卷积，因此每个数字就代表一个特征的强度，而Gram计算的实际上是两两特征之间的相关性，哪两个特征是同时出现的，哪两个是此消彼长的等等，同时，Gram的对角线元素，还体现了每个特征在图像中出现的量，因此，Gram有助于把握整个图像的大体风格。有了表示风格的Gram Matrix，要度量两个图像风格的差异，只需比较他们Gram Matrix的差异即可。

来自知乎阳仔的下午茶：gram矩阵用在风格迁移的话，举个简单的例子吧，我是这么理解的：

比如有一张写实图片，内容是狗坐在草地上，如果单看这张图片，两个毫不相干的物体（例如狗的头和草地上的一棵草的联系，对应gram矩阵的话就是feature map中狗头的channel和草的channel的内积）我们是看不出来有什么联系的，事实上gram矩阵中大部分的元素都是这样的（很多元素是人类看起来毫不相干的两个channel特征的内积）。但是如果有一张卡通图片，也是狗在草地上，这时刚刚写实图片里狗头和草地上的草的联系我们就有所对照，就可以得出一个结论了，也即他们的联系就是他们都是写实的，而在卡通图上面他们的联系就是他们都是卡通的。

也就是说这种联系就只能是风格了，所以如果能使得两张图片的gram矩阵一致，应该是可以使得其风格大概一致的。

来自CSDN Sun7_She

gram矩阵是计算每个通道I的feature map与每个通道j的feature map的内积。

gram matrix的每个值可以说是代表i通道的feature map与j通道的feature map的互相关程度。

一言以蔽之：格拉姆矩阵抽取到的是：在图片同一位置下，不同特征之间的组合。

#### 总结常用的损失函数

- #### Content Loss

  Content Loss衡量content activation map与原始content image之间的差别

  $L_{c} = w_{c}*\sum_{i,j}(F_{ij} - P_{ij})$

  ```python
  def content_loss(weight, current, original):
      loss = weight * tf.reduce_sum((current - original)**2)
      return loss
  ```

  因为我们要计算feature之间的误差，所以直接计算了feature的L2 Loss.

- #### Style Loss

  我们要如何衡量生成图片与style 图片之间在style上的差别呢？ 这是机器学习的一个很重要的思想， 使用一些数学参数来衡量我们想要衡量的指标， 从而产生一个可优化的Loss。我们用图像feature map的feature neuron之间的相关性来代表图像的“风格”， 那么， 生成图片与style image之间的“风格”差异就可以用所有像素之间的相关性值的L2 Loss来表示。如何表达所有像素两两之间的相关性呢， 这里， 我们用feature map的neuron之间的correlation相关系数来表达这种误差， 称之为gram map. 对于任意两个通道， 我们计算这两个通道之间每个元素的乘积(C*C)， 这代表了这两个通道之间的相关性， 共有H\*W这么多个这样的C\*C矩阵， 求这些矩阵的均值， 就代表了两张图所有通道之间的相关性gram map.

  ```python
  def gram_matrix(features, normalize=True):
      shape = tf.shape(features)
      features_reshaped = tf.reshape(features, (shape[1]*shape[2], shape[3]))
      gram = tf.matmul(tf.transpose(features_reshaped), features_reshaped)
      if normalize:
          gram /= tf.cast((shape[3] * shape[1] * shape[2]), tf.float32)
      return gram
  
  def style_loss(feats, style_layers, style_targets):
      total_loss = 0.0
      for i in range(len(style_layers)):
          G = style_targets[i]
          A = gram_matrix(feats[style_layers[i]])
          total_loss += tf.reduce_sum((G - A)**2)
      return total_loss
  ```

- #### TV LOSS

  有了Style Loss 和 Content Loss， 分别代表生成图片与style 图片之间的“风格差异”， 以及生成图片与content 图片之间的“feature差异”。 理论上， 我们就可以得到一个融合了风格图片风格的内容图片。

  但这里存在过拟合。我们需要加入regularization项， 如何理解这种regularization呢？ 我们认为， 过度拟合就是图像为了拟合风格/内容， 而在图片的相邻像素之间， 产生了巨大的差异。 使得图片“看起来不自然”， 因此我们直接用相邻像素之间的差做regularization项。

  ```python
  def tv_loss(img, tv_weight):
      # Your implementation should be vectorized and not require any loops!   
      left_loss = tf.reduce_sum((img[:, 1:, :, :] - img[:, :-1, :, :])**2)
      down_loss = tf.reduce_sum((img[:, :, 1:, :] - img[:, :, :-1, :])**2)
      loss = tv_weight*(left_loss + down_loss)
      return loss
  ```

### 论文笔记

> Texture Synthesis Using CNN

- #### Introduction

  视觉纹理合成的目标是从一个样本纹理中推导一个泛化的过程，用来生成具有那种纹理的任意的新图像。合成的纹理的质量评判标准通常是人工检验，如果人们无法分辨合成纹理，纹理就是成功合成的。

  一般有两种纹理生成方法。第一种方法是，对像素或原始纹理的整个区块再采样生成新纹理。这些非参数再采样技术和大量扩展或改进方法可以非常有效地生成高质量的自然纹理。然而，它们无法定义为自然纹理定义一个真正的模型，但只能提供一个机械式的过程，随机选择一个源纹理，不改变它的感知特点。

  相反，第二种纹理合成的方法是明确定义一个参数化的纹理模型。模型通常由一组图像空间统计值组成。在模型中，纹理由这些观测值的结果唯一确定，每个生成相同结果的图像应该具有相同的纹理。因此，一个纹理的新样本可以通过寻找生成与原始纹理相同观测结果的图像来生成。这个思路是Julesz在论文13中第一次提出，他推测视觉纹理可以用像素的N阶直方图唯一地描述。后来，受哺乳动物早期的视觉系统的线性响应特性的启发的纹理模型，与方向带通滤波（Gabor）非常相似。这些纹理模型基于滤波响应的统计测量值，而不是直接作用在图像像素上。目前，最好的纹理合成的参数模型，可能是论文21中的，计算一个称为方向金字塔（论文24）的线性滤波响应池的手工统计数据。然而，尽管这个模型在比较宽的纹理范围上合成性能非常好，但无法捕捉自然纹理的所有范围。

  本文提出了一种新的参数纹理模型处理这个问题，如图1所示。我们并不是基于早期的视觉系统模型来描述纹理，而是采用了卷积神经网络，整个腹流功能模型，作为我们纹理模型的基础。我们合并了特征响应的空间统计结构框架，强大的卷积神经网络特征空间在物体识别上训练好了。卷积神经网络的层级处理架构上构建的空间不变性表示，对纹理模型进行参数化。

  ![method.png](https://github.com/Gary11111/05-Texture/blob/master/img/method.png?raw=true)

  合成方法。纹理分析（左边）。原始纹理传入卷积神经网络，计算大量层上的特征响应克莱姆矩阵Gl。纹理合成（右边）。白噪声图像$\hat{x}$传入卷积神经网络，计算包含纹理模型的每个层的损失函数El。在每个像素值的总损失函数上计算梯度下降，生成与原始纹理相同的克莱姆矩阵$\hat{G_{l}}$得到一个新图像。

- #### Texture Generation

  为了生成给定图像的新纹理，使用白噪声图像的梯度下降查找另外一个图像，可以匹配原始图像的克莱姆矩阵表示。这个优化可以用原始整体图像的克莱姆矩阵和生成图像的克莱姆矩阵之间的平均平方距离的最小化来处理:
  $$
  E_{l} = \frac{1}{4N_{l}^{2}M_{l}^2}\sum_{i,j}(G_{ij}^{l} - \hat{G_{ij}^{l}})
  $$
  总的损失函数为：
  $$
  L(x,\hat{x}) = \sum_{l=0}^{L}w_{l}E_{l}
  $$
  本文使用L-BFGS算法，对于高维度优化问题看起来是比较合理的选择。整个流程主要依赖于用于训练卷积网络的标准的前馈-反馈传递。因此，尽管模型比较复杂，纹理生成仍然可以在使用GPU和训练深度神经网络的优化工具时在合理的时间内完成。

### 实验任务

- 实现格拉姆矩阵和损失函数

  ```python
  def gram_matrix(feature_map, normalize=True):  # 获得gram矩阵
      shape = tf.shape(feature_map)
      """三维矩阵相乘的技巧"""
      features_reshaped = tf.reshape(feature_map, (shape[1] * shape[2], shape[3]))  # 展成二维
      gram = tf.matmul(tf.transpose(features_reshaped), features_reshaped)  # 乘以转置得到channel*channel的矩阵
      if normalize:
          gram /= tf.cast((shape[3] * shape[1] * shape[2]), tf.float32)  # 标准化
      return gram
  
  
  def get_gram_loss(noise_layer, source_layer):
      """layer中包含所有的feature map each layer's shape: (batch,w,h,channel)"""
      noise_gram = gram_matrix(noise_layer)
      source_gram = gram_matrix(source_layer)  # gram矩阵的维度为channel*channel
      gram_loss = tf.reduce_sum((source_gram - noise_gram) ** 2)  # 得到一个value
      # print("gram_loss === > ", gram_loss)
      return gram_loss
  ```

  结果如下：

  ![q1.png](https://github.com/Gary11111/05-Texture/blob/master/img/q1.png?raw=true)

- 用非纹理图片训练

- 使用更少的层训练，保持格拉姆矩阵

  生成过程模拟。每排对应网络不同的处理阶段。只有在最低层约束纹理表示，合成的纹理有很少的结构，与光谱噪声非常相似（第一排）。随着匹配纹理表示的网络层数增加，生成图像的自然度也增加（第2-5排；左边的标签指的是包含的最上层）。前3列的源纹理采用的是论文21的。为了更好地对比，在最后一排现实了他们的结果。最后一列显示的非纹理图像生成的纹理，为了更直觉地表现纹理模型是如何表示图像信息的。

  ![all.PNG](https://github.com/Gary11111/05-Texture/blob/master/img/all.PNG?raw=true)

- 找到格拉姆矩阵的替代方案 `EMD`

  给定两个签名(或者叫分布、特征量集合)和，为个特征量和其权重的集合，记作，即上图左侧部分。同样的，还有一个分布，，即上图右侧部分。在计算这个这两个签名的Earth Mover’s Distance(EMD)[5]前，我们要先定义好、中任意取一个特征量( and )之间的距离（这个距离叫ground distance，两个签名之间EMD依赖于签名中特征量之间的ground distance）。当这两个特征量是向量时得到的是欧式距离，当这两个特征量是概率分布时得到的是相对熵(KL距离/Kullback–Leibler divergence)。现在，给定两个签名(和)，只要计算好每两个特征量之间的距离，系统就能给出这两个签名之间的距离了。

  在计算机科学与技术中，地球移动距离(EMD)是一种在D区域两个概率分布距离的度量，就是被熟知的Wasserstein度量标准。不正式的说，如果两个分布被看作在D区域上两种不同方式堆积一定数量的山堆，那么EMD就是把一堆变成另一堆所需要移动单位小块最小的距离之和。上述的定义如果两个分布有着同样的整体（粗浅的说，就像两个堆有着同样的数量），在规范化的直方图或者概率密度函数上。在这基础上，EMD等同于两个分布的第一Mallows距离或者第一Wasserstein距离。

  不同情况下EMD使用方式也不一样，但还是有一些共通之处。比如权重都是指特征量的重要程度。例如，一个直方图对应一个签名的情况下，直方图中的每一根柱(bar)代表一个特征量，柱的高度就对应其权重。在之前的相似图像检索(2009/10/3）一文中，我使用到了图像颜色分布直方图相交距离(Histogram Intersection )，也可以用在EMD中当作ground distance使用。最早提出EMD概念的论文中有提到，EMD最初就是用来做相似图片检索的。

  按照论文的要求，EMD的距离公式表示如下：
  $$
  Loss_{emd} = \sum_{l}w_{l}\sum_{i}(sorted(F_{i})-sorted(\hat{F_{i}}))
  $$
  python 实现如下：

  ```python
  def get_EMD_loss(noise_layer, source_layer):
      shape = tf.shape(noise_layer)
      """三维矩阵相乘的技巧"""
      noise_reshaped = tf.reshape(noise_layer, (shape[1] * shape[2], shape[3]))  # 展成二维: 每一行是一个feature map
      source_reshaped = tf.reshape(source_layer, (shape[1] * shape[2], shape[3]))
      noise_sort = tf.sort(noise_reshaped, direction='ASCENDING')
      source_sort = tf.sort(source_reshaped, direction='ASCENDING')
      return tf.reduce_sum(tf.math.square(noise_sort - source_sort))
  ```

  ![EMD.PNG](https://github.com/Gary11111/05-Texture/blob/master/img/EMD.PNG?raw=true)

  从EDM的结果来看，EDM会可以学习到图像feature-map之间的相关性，但是过滤了很多颜色特征，或许是因为EDM对feature map进行了sort排序，打乱了原图像素的排序。
  另外：在加入了L2正则化项之后，图像颜色有了恢复，并且形状更加接近原图的辣椒形状。进一步在L2的基础上加入TV-loss图像的颜色恢复了，并且辣椒的形状也得到保护。

- 改变权重因子。

  改变各层的权重：1. 权重向深层递增；2. 权重向深层递减。
  
  ![weights.PNG](https://github.com/Gary11111/05-Texture/blob/master/img/weights.PNG?raw=true)
  
  

### 实验中遇到的困难

- 计算gram矩阵损失的时候，矩阵的维数不匹配。

  ```
  ValueError: Dimension 1 in both shapes must be equal, but are 28 and 14. Shapes are [1,28,28,512] and [1,14,14,512].
  From merging shape 9 with other shapes. for 'packed' (op: 'Pack') with input shapes: [1,224,224,64], [1,224,224,64], [1,112,112,128], [1,112,112,128], [1,56,56,256], [1,56,56,256], [1,56,56,256], [1,28,28,512], [1,28,28,512], [1,28,28,512], [1,14,14,512], [1,14,14,512], [1,14,14,512].
  ```

  可以发现，这些对应于vgg16每一个卷积层的feature map的维度。并且可以看到随着层数的加深，feature map的大小越来越小，但是通道数越来越多，说明最后神经网络已经把提取到的特征分散保存到不同的通道中，即：每一个通道都把注意力放在自己应该关注的点上。这与filter的功能是对应起来的，每一个filter只关注与自己扫过的区域内重要的特征，可以理解为不同的参数提取出不同的特征。

  解决方法：

  在设计损失函数的时候需要对feature map做矩阵乘法，feature map第一维默认是batchsize，如果遗漏该维度的话，则会出现上述情况。

- 在计算gram loss时，损失恒为0.

  原因：gram loss进行了两次标准化。

  并且，，gram loss的值很小，需要保留到小数点后10位，因为图像是归一化的。