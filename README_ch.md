简体中文 | [English](README.md)
<div align=center><img src="doc/pic/OMML logo.png" width="500px;" /></div>

<a href="./LICENSE"><img src="https://img.shields.io/badge/license-Apache%202-dfd.svg"></a>
<a href=""><img src="https://img.shields.io/badge/python-3.6+-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/paddlepaddle-2.1.3-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/torch-1.7.1-aff.svg"></a>
<a href=""><img src="https://img.shields.io/badge/os-linux-pink.svg"></a>

## 简介
多模态学习工具包OMML 以**PyTorch**为主，兼容 **Paddle**版本，可通过更换branch方式进行使用，旨在于提供模态联合学习和跨模态学习算法模型库，为处理图片文本等多模态数据提供高效的解决方案，助力多模态学习应用落地。

OMML 发布作者：
- 南京理工大学 KMG 小组


|                  杨杨                   |   郭伟立   |          |
| :-------------------------------------: | :----------: | :--------------: |
|    鲍然   |     张宇萱     | 奚文涓 |





### 近期更新
- Add visualization TSNE of fusion task 2023.1.14
- Add model DOMFN 2022.11.17
- Add model CPRC 2022.11.14
- Add model TMC 2022.5.8

[更多](doc/update.md)

## 特性
- 丰富的任务场景：工具包提供多模态融合、跨模态检索、图文生成等多种多模态学习任务算法模型库，支持用户自定义数据和训练。
- 成功的工业应用：基于工具包算法已有相关落地应用，如球鞋真伪鉴定、图像字幕生成、舆情监控等。


### 应用展示
- 球鞋真伪鉴定 (更多信息欢迎访问我们的网站 [Ysneaker](http://www.ysneaker.com/) ！)

<div align=center><img src="doc/pic/identify.gif" width="600px;" /></div>


- 更多应用

<div align=center><img src="doc/pic/app.png" width="600px;" /></div>


### 落地实践
- 与百度人才智库（TIC）合作 [智能招聘](https://ai.baidu.com/solution/recruitment) 简历分析，基于多模态融合算法成功落地。

<div align=center><img src="doc/pic/tic.png" width="600px;" /></div>

### 可视化模块
- 特征层面：<br>
  举例参数设置：dataset="twitter", data_mode="twitter", visual="tsne", choose="fusion"<br>
  文件输出目录：out_root/tsne
  <div align=center><img src="doc/pic/visual.jpg" width="600px;" /></div>

- 模型层面：<br>
- 指标层面：<br>


## 框架
OMML 包括torch 版本 torchmm 和 paddle 版本 paddlemm 包，由以下三个模块组成：
- 数据处理：提供统一的数据接口和多种数据处理格式
- 模型库：包括多模态融合、跨模态检索、图文生成、多任务算法
- 训练器：对每种任务设置统一的训练流程和相关指标计算

<div align=center><img src="doc/pic/framework.png" width="300px;" /></div>

### 使用
下载工具包

```
git clone https://github.com/njustkmg/OMML.git
```

- 数据搭建说明 [教程](data/README.md)
- 依赖文件下载 [教程](paddlemm/metrics/README.md) 

#### OMML 使用示例：
##### torchmm 使用示例：

```python
from torchmm import TorchMM

# config: Model running parameters, see configs/
# data_root: Path to dataset
# image_root: Path to images
# cuda: Which gpu to use

runner = TorchMM(config='configs/cmml.yml',
                 data_root='data/COCO', 
                 image_root='data/COCO/images',
                 out_root='experiment/cmml_torch',
                 cuda=0)

runner.train()
runner.test()
```

或者

```
python run_torch.py --config configs/cmml.yml --data_root data/COCO --image_root data/COCO/images --out_root experiment/cmml_torch --cuda 0
```
##### paddlemm 使用示例：
```python
from paddlemm import PaddleMM

# config: Model running parameters, see configs/
# data_root: Path to dataset
# image_root: Path to images
# gpu: Which gpu to use

runner = PaddleMM(config='configs/cmml.yml',
                  data_root='data/COCO', 
                  image_root='data/COCO/images', 
                  out_root='experiment/cmml_paddle',
                  gpu=0)

runner.train()
runner.test()
```

或者

```
python run.py --config configs/cmml.yml --data_root data/COCO --image_root data/COCO/images --out_root experiment/cmml_paddle --gpu 0
```



### 模型库 (持续更新中)

<div align=center><img src="doc/pic/classes.png" width="500px;" /></div>

- 模态联合学习-融合学习
  - Early (Multi-modal early fusion)
  - Late (Multi-modal late fusion)
  - LMF ([Efficient Low-rank Multimodal Fusion with Modality-Specific Factors](https://arxiv.org/pdf/1806.00064.pdf))
  - TMC ([Trusted Multi-View Classification](https://arxiv.org/pdf/2102.02051))
- 模态联合学习-协同训练
  - CMML ([Comprehensive Semi-Supervised Multi-Modal Learning](https://www.ijcai.org/proceedings/2019/0568.pdf))
- 跨模态学习-模态翻译
  - ShowAttendTell ([Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf))
  - AoANet ([Attention on Attention for Image Captioning](https://arxiv.org/pdf/1908.06954.pdf))
  - CPRC ([Exploiting Cross-Modal Prediction and Relation Consistency for Semi-Supervised Image Captioning](https://arxiv.org/pdf/2110.11767v2.pdf))
- 跨模态学习-模态对齐
  - VSE++ ([VSE++: Improving Visual-Semantic Embeddings with Hard Negatives](https://arxiv.org/pdf/1707.05612.pdf))
  - SCAN ([Stacked Cross Attention for Image-Text Matching](https://arxiv.org/pdf/1803.08024.pdf))
  - BFAN ([Focus Your Attention: A Bidirectional Focal Attention Network for Image-Text Matching](https://dl.acm.org/doi/10.1145/3343031.3350869))
  - IMRAM ([IMRAM: Iterative Matching with Recurrent Attention Memory for Cross-Modal Image-Text Retrieval](https://arxiv.org/pdf/2003.03772.pdf))
  - SGRAF ([Similarity Reasoning and Filtration for Image-Text Matching](https://arxiv.org/pdf/2101.01368.pdf))
- 基于 Transformer 结构的多任务框架
  - VILBERT ([ViLBERT: Pretraining Task-Agnostic Visiolinguistic Representations for Vision-and-Language Tasks](https://arxiv.org/pdf/1908.02265.pdf))
  

## 技术支撑

### 多模态论文

- Chuan Qin, Hengshu Zhu, Tong Xu, Chen Zhu, Liang Jiang, Enhong Chen, Hui Xiong, Enhancing Person-Job Fit for Talent Recruitment: An Ability-aware Neural Network Approach, In Proceedings of the 41st International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR-2018) , Ann Arbor, Michigan, USA, 2018.
- Chen Zhu, Hengshu Zhu, Hui Xiong, Chao Ma, Fang Xie, Pengliang Ding, Pan Li, Person-Job Fit: Adapting the Right Talent for the Right Job with Joint Representation Learning, In ACM Transactions on Management Information Systems (ACM TMIS), 2018.
- Dazhong Shen, Hengshu Zhu, Chuan Qin, Tong Xu, Enhong Chen, Hui Xiong, Joint Representation Learning with Relation-enhanced Topic Models for Intelligent Job Interview Assessment, In ACM Transactions on Information Systems (ACM TOIS) , 2021.
- Yang Yang, Hong-Chen Wei, Heng-Shu Zhu, Dian-Hai Yu, Hui Xiong, Jian Yang. Exploiting Cross-Modal Prediction and Relation Consistency for Semi-Supervised Image Captioning. **IEEE Transactions on Cybernetics (IEEE TCYB)**, 2022 in press. (CCF-B).[[Pytorch Code\] ](https://github.com/njustkmg/CPRC-Pytorch)[[Paddle Code\]](https://github.com/njustkmg/CPRC-Paddle)
- Yang Yang, Jia-Qi Yang, Ran Bao, De-Chuan Zhan, Hengshu Zhu, Xiao-Ru Gao, Hui Xiong, Jian Yang. Corporate Relative Valuation using Heterogeneous Multi-Modal Graph Neural Network. IEEE Transactions on Knowledge and Data Engineering (IEEE TKDE), 2021. (CCF-A). [Code](https://github.com/njustkmg/TKDE21_HMM)
- Yang Yang, De-Chuan Zhan, Yi-Feng Wu, Zhi-Bin Liu, Hui Xiong, and Yuan Jiang. Semi-Supervised Multi-Modal Clustering and Classification with Incomplete Modalities. IEEE Transactions on Knowledge and Data Engineering (IEEE TKDE), 2020. (CCF-A)
- Yang Yang, Chubing Zhang, Yi-Chu Xu, Dianhai Yu, De-Chuan Zhan, Jian Yang. Rethinking Label-Wise Cross-Modal Retrieval from A Semantic Sharing Perspective. Proceedings of the 30th International Joint Conference on Artificial Intelligence (IJCAI-2021), Montreal, Canada, 2021. (CCF-A).
- Yang Yang, Yi-Feng Wu, De-Chuan Zhan, Zhi-Bin Liu, Yuan Jiang. Complex Object Classification: A Multi-Modal Multi-Instance Multi-Label Deep Network with Optimal Transport. Proceedings of the Annual Conference on ACM SIGKDD (KDD-2018) , London, UK, 2018. [Code](https://github.com/njustkmg/KDD18_M3DN)
- Yang Yang, Jingshuai Zhang, Fan Gao, Xiaoru Gao, Hengshu Zhu DOMFN: A Divergence-Orientated Multi-Modal Fusion Network for Resume Assessment. ACM MULTIMEDIA(ACMMM-2022) , Lisbon, Portugal, 2022.(CCF-A)
- 更多论文欢迎访问我们的网站 [njustkmg](http://www.njustkmg.cn/) ！

### 飞桨论文复现挑战赛

- 飞桨论文复现挑战赛 (第四期)：《Comprehensive Semi-Supervised Multi-Modal Learning》赛题冠军
- [飞桨论文复现挑战赛 (第五期)](https://aistudio.baidu.com/aistudio/competition/detail/126/0/introduction)：《From Recognition to Cognition: Visual Commonsense Reasoning》赛题冠军
- [飞桨论文复现挑战赛(第六期)](https://aistudio.baidu.com/aistudio/competition/detail/205/0/task-definition): 《Efficient Low-rank Multimodal Fusion with Modality-Specific Factors》赛题冠军



## 贡献
- PaddlePaddle 复现代码问题记录 [链接](doc/paddle.md) 。
- 我们非常欢迎您为 OMML 贡献代码，也十分感谢你的反馈。


## 许可证书
本项目的发布受 <a href="LICENSE">Apache 2.0 license</a> 许可认证。
