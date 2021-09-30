# R-Drop: Regularized Dropout for Neural Networks
##  一、简介
### 摘要：  
&emsp;&emsp;Dropout是一种强大且被大家广泛使用的深度学习技术，它通常被用来正则化深度神经网络的训练，抑制过拟合。在这篇论文，作者介绍了一种使用于模型训练过程中的基于dropout的简单的正则化策略，  这个策略强迫通过dropout产生的不同子模型的输出分布随着训练的进行逐渐地变得互相一致。具体而言，对于每一个训练样本，R-Drop最小化通过dropout随机采样得到的两个子模型的输出分布之间的双向  KL散度。以[Vit](https://link.zhihu.com/?target=http%3A//arxiv.org/abs/2010.11929)为例，R-Drop的整体框架如下图所示：  
![](https://github.com/zpc-666/Paddle-R-Drop/blob/main/images/930.PNG)  
该图左边展示的是一个输入x输入模型两次，得到两个输出分布P1、P2，由于dropout的存在，这个过程可以展开为该图右边的结构，也就是输入x两次输入到的是由dropout随机生成的两个不同的子模型，  通过利用双向KL散度来使得这两个子模型的输出分布互相一致，从而达到正则化的目的，这种应该算网络扰动吧。 实现起来很简单，以下参考自[官方repo](https://github.com/dropreg/R-Drop)的Readme.md，但他在vit部分代码中并不是把R-drop策略以以下方式嵌入，而是将kl散度的计算放在了Vit模型的前向传播函数中，但效果其实是一样的，我们的复现是参考后者，详情见models文件夹的modeling.py的279~308行。
```
import paddle.nn.functional as F

# define your task model, which outputs the classifier logits
model = TaskModel()

def compute_kl_loss(self, p, q):
    
    p_loss = F.kl_div(F.log_softmax(p, axis=-1), F.softmax(q, axis=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, axis=-1), F.softmax(p, axis=-1), reduction='none')

    # You can choose whether to use function "sum" and "mean" depending on your task
    p_loss = p_loss.sum()
    q_loss = q_loss.sum()

    loss = (p_loss + q_loss) / 2
    return loss

# keep dropout and forward twice
logits = model(x)

logits2 = model(x)

# cross entropy loss for classifier
ce_loss = 0.5 * (cross_entropy_loss(logits, label) + cross_entropy_loss(logits2, label))

kl_loss = compute_kl_loss(logits, logits2)

# carefully choose hyper-parameters
loss = ce_loss + α * kl_loss
```
&emsp;&emsp; 本项目是基于R-drop的基于cifar100数据集的ViT-B/16模型在Paddle 2.x上的开源实现。论文效果图与复现指标：  
![](https://github.com/zpc-666/Paddle-R-Drop/blob/main/images/9302.PNG)

## 二、复现精度
&emsp;&emsp; 本次比赛的验收标准： CIFAR-100 ViT-B/16+RD=93.29 （论文指标）。我们的复现结果对比如下所示：
<table>
    <thead>
        <tr>
            <th>来源</th>
            <th>test acc</th>
            <th>模型权重</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>原论文实现（指标）</td>
            <td>93.29%</td>
            <td>https://aistudio.baidu.com/aistudio/datasetdetail/104562</td>
        </tr>
        <tr>
            <td>paddle实现</td>
            <td>92.71%、92.65%、92.66%</td>
            <td>https://pan.baidu.com/s/1KIh2cU4boC7sxisr4IRJdA </td>
          </tr>
    </tbody>
</table>

模型权重提取码：zpc6

几次尝试没搞懂脚本运行，只能跑单卡，10000步大概V100单卡跑2天6小时26分，200000步要跑好久好久，比赛结束都跑不完，而且也没有上千算力去烧，最终将参数训练步数设为10000或20000。在8月17至9月12日训练、调参，两个号共耗费400多小时算力，最终也没跑出指标，估计还是总step数目和跑的step数目不够。hhh,在总训练步数10000或20000下跑的结果，一度让我怀疑我没用dropout或者实现有问题，但核心代码完全按官方repo改为paddle，只是一些与模型训练无关的输出格式和保存、组织结构上改成了自己舒服的风格，且模型已完成了与pytorch版本的假数据前向对齐，最最最重要的是我单卡跑pytorch版本的，在配置参数一样下，输出精度是几乎一样的。
## 三、数据集
根据复现要求我们用的是 Cifar100 数据集。  
* 数据集大小：100类别，训练集有50000张图片。测试集有10000张图片，图像大小为32x32，彩色图像；
* 数据格式：用paddle.vision.datasets.Cifar100调用，格式为cifar-100-python.tar.gz

## 四、环境依赖
硬件：使用了百度AI Studio平台的至尊GPU  
框架：PaddlePaddle >= 2.0.0，平台提供了除ml_collections以外的所有依赖。可以通过`pip install ml_collections`进行下载安装。

## 五、快速开始
&emsp;&emsp;R_Drop/configs/configs.py中提供了R-Drop和Vit论文以及官方repo中参数的默认配置，故以下只按默认配置指导如何使用，如需修改参数可以直接在configs.py中修改，或按argparse的用法显式地修改相应参数。
1. 半精度混合训练（推荐）
```
%cd R_Drop/
!python main.py --n_gpu 1 --name Cifar100-ViT_B16 --gradient_accumulation_steps 32 --fp16
```
2. 普通训练
```
%cd R_Drop/
!python main.py --n_gpu 1 --name Cifar100-ViT_B16 --gradient_accumulation_steps 32
```
3. 评估
```
%cd R_Drop/
!python main.py --n_gpu 1 --name Cifar100-ViT_B16 --mode eval --checkpoint_dir output/Cifar100-ViT_B16_checkpoint.pdparams
```
4. 恢复训练（官方repo没有提供）
```
%cd R_Drop/
!python main.py --n_gpu 1 --name Cifar100-ViT_B16 --gradient_accumulation_steps 32 --fp16 --resume output/
```
## 六、代码结构与详细说明
&emsp;&emsp; 几乎完全参考[Contrib 代码提交规范](https://github.com/PaddlePaddle/Contrib/wiki/Contrib-%E4%BB%A3%E7%A0%81%E6%8F%90%E4%BA%A4%E8%A7%84%E8%8C%83) ，  
```
./R_Drop  
|-- configs   # 参数配置文件夹  
|-- data     # 数据文件夹  
|-- models   # 模型实现文件夹  
|-- utils   # 工具类API文件夹  
|- checkpoint	# imagenet21k pre-train文件夹  
|- output 	# 模型保存文件夹  
|- logs		# 训练日志文件夹  
|-- evaluate.py  # 执行评估功能的代码  
|-- main.py  # 主函数，负责调用所有功能  
|-- run.sh   # 运行脚本，需要提供不同环境下运行的示例  
|- train.py  # 执行训练功能的代码  
参数详解见config.py每个参数的help信息。
```
## 七、 参考
  几乎完全按照官方repo实现来写
  * 官方实现 https://github.com/dropreg/R-Drop/tree/main/vit_src
  * 论文 https://arxiv.org/pdf/2106.14448v1.pdf
