# replknet 代码解析
## torch库函数
- ***Conv2d***
> 卷积层
```properties
class Conv2d(
    in_channels: int,
    out_channels: int,
    kernel_size: _size_2_t,
    stride: _size_2_t = 1,
    padding: _size_2_t | str = 0,
    dilation: _size_2_t = 1,
    groups: int = 1,
    bias: bool = True,
    padding_mode: str = "zeros",
    device: Any | None = None,
    dtype: Any | None = None
)
Args:
    in_channels (int): Number of channels in the input image
    out_channels (int): Number of channels produced by the convolution
    kernel_size (int or tuple): Size of the convolving kernel
    stride (int or tuple, optional): Stride of the convolution. Default: 1
    padding (int, tuple or str, optional): Padding added to all four sides of
        the input. Default: 0
    dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
    groups (int, optional): Number of blocked connections from input
        channels to output channels. Default: 1
    bias (bool, optional): If ``True``, adds a learnable bias to the
        output. Default: ``True``
    padding_mode (str, optional): ``'zeros'``, ``'reflect'``,
        ``'replicate'`` or ``'circular'``. Default: ``'zeros'``
```

- ***DepthWiseConv2dImplicitGEMM***
> 基于[深度可分离卷积](https://blog.csdn.net/m0_37605642/article/details/134174749)实现的高性能计算库由于代替Conv2d
```properties
class DepthWiseConv2dImplicitGEMM(
    channels: Any,
    kernel: int | Any,
    bias: bool = False
)
```

- ***BatchNorm2d***
> 用于对数据归一化处理,避免导致Relu数据过大而不稳定
```properties
class BatchNorm2d(
    num_features: int,
    eps: float = 0.00001,
    momentum: float | None = 0.1,
    affine: bool = True,
    track_running_stats: bool = True,
    device: Any | None = None,
    dtype: Any | None = None
)
```
$$y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta
$$
> Args:  
    num_features: :math:`C` from an expected input of size
        :math:`(N, C, H, W)`  
    eps: a value added to the denominator for numerical stability.
        Default: 1e-5  
    momentum: the value used for the running_mean and running_var
        computation. Can be set to ``None`` for cumulative moving average
        (i.e. simple average). Default: 0.1  
        默认的 momentum = 0.1，表示“新值权重占10%，旧值保留90%”
    affine: a boolean value that when set to ``True``, this module has
        learnable affine parameters. Default: ``True``  
    track_running_stats: a boolean value that when set to ``True``, this
        module tracks the running mean and variance, and when set to ``False``,
        this module does not track such statistics, and initializes statistics
        buffers :attr:`running_mean` and :attr:`running_var` as ``None``.
        When these buffers are ``None``, this module always uses batch statistics.
        in both training and eval modes. Default: ``True``

- ***ReLU***
$$\text{ReLU}(x) = (x)^+ = \max(0, x)$$

- ***GELU***
$$\text{GELU}(x) = x * \Phi(x)$$
$$\text{GELU}(x) = 0.5 * x * (1 + \text{Tanh}(\sqrt{2 / \pi} * (x + 0.044715 * x^3)))$$

## 自定义函数与类
- ***fuse_bn***
> 参考  
[网络inference加速：Fuse Conv&BN](https://blog.csdn.net/qq_42191914/article/details/103306066)  
在网络的推理阶段，可以将BN层的运算融合到Conv层中，减少运算量，加速推理。本质上是修改了卷积核的参数，在不增加Conv层计算量的同时，略去了BN层的计算量

考虑只使用BN的偏置项,Conv2d则不使用偏置项,避免冗余,在公式推导时则暂且加上  
Conv2d参数:$\omega$,$b$  
BN参数:$\gamma$,$\sigma$,$\epsilon$,$\beta$  
对于输入$x$,则$x\to Conv2d \to BN$具体过程如下
$$
x_1=\omega \times x+ b
\tag{1}
$$
$$
x_2=\gamma \times \frac{x_1-\mu}{\sqrt{\sigma ^2+\epsilon}}+\beta
\tag{2}
$$
将(1)代入(2),得到
$$
x_2=\gamma \times \frac{\omega \times x+ b-\mu}{\sqrt{\sigma ^2+\epsilon}}+\beta \\
= \underbrace{\frac{\gamma \times \omega}{\sqrt{\sigma ^2+\epsilon}}}_{\omega_1}\times x + \underbrace{\gamma \times \frac{b-\mu}{\sqrt{\sigma ^2+\epsilon}} + \beta}_{b_1}
\tag{3}
$$
即(3)最终化简为与(1)相同结构  
最终实现时不使用$b$

- (class) ReparamLargeKernelConv
> 将大卷积核与小卷积核结合并行训练,关键在于设置大卷积核与小卷积核的填充确保两者输出特征图维度相同,同时也实现大小卷积核融合为一个等效大卷积核,减少参数量

根据输出特征图与输入特征图关系
$$
output\_size=\frac{input\_size+2\times padding-kernel\_size}{stride}+1
$$

可以设置padding确保输入输出特征图都相同
$$
padding=[\frac{kernel\_size}{2}]
$$

大小卷积核融合是将小卷积核置于中心,并使用零填充使之与大卷积核形状一直再相加操作

- (class) ConvFFN
> 卷积前馈网络模块