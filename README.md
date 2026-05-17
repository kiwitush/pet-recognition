# 微调在 ImageNet 上预训练的卷积神经网络实现宠物识别

使用 **Oxford-IIIT Pet Dataset** 对 ImageNet 预训练的 ResNet 进行微调，实现 37 种宠物品种分类。

---

## 项目结构

```
pet_recognition/
├── config.py                  # 集中超参数配置
├── dataset.py                 # 数据加载与预处理
├── trainer.py                 # 训练引擎 
├── models/
│   ├── baseline.py            # ResNet-18/34 微调
│   ├── att_block.py            # SE-block / CBAM 注意力模块
│   ├── attention.py            # ResNet + 注意力机制的进阶模型
│   └── vit.py                  # ViT-Tiny/Small 模型定义
├── experiments/
│   ├── train_baseline.py      # Baseline 训练 
│   ├── train_attention.py     # 引入注意力机制的模型训练 (ResNet + SE/CBAM)
│   ├── train_contrast.py      # 预训练消融实验 (随机初始化 vs 预训练)
│   ├── train_vit.py           # ViT-Tiny/Small 训练
│   └── hyperparameter_search.py  # 超参数网格搜索 
├── inference.py               # 推理: 单张图片预测 / 测试集评估         
└── requirements.txt
```



## 数据集

**Oxford-IIIT Pet Dataset** (37 类猫狗品种)，首次运行时会自动下载到 `./data/` 目录，并进行数据预处理与增强

```bash
python dataset.py
```


## 实验运行

### Baseline (预训练 ResNet-18 微调)

```bash
python experiments/train_baseline.py

# 可通过命令行覆盖参数
python experiments/train_baseline.py lr=1e-5 epochs=50 batch_size=64 model_name=resnet34
```

### 超参数搜索

对学习率、训练轮数、优化器进行网格搜索，找出最优组合

```bash
python experiments/hyperparameter_search.py
```

### 预训练消融实验

随机初始化训练 vs 预训练微调，观察预训练带来的提升

```bash
python experiments/train_contrast.py
```

### 引入注意力机制

```bash
# CBAM 注意力 (默认)
python experiments/train_attention.py

# SE 注意力
python experiments/train_attention.py attention_type=se
```

### ViT 

```bash
# ViT-Tiny (默认)
python experiments/train_vit.py

# ViT-Small
python experiments/train_vit.py model_name=vit_small
```


## 模型权重

训练好的模型保存在 `./checkpoints/<experiment_name>/best.pth`。


## 推理

```bash
# 单张图片预测，默认 ResNet-18
python inference.py path/to/pet_image.jpg

# 指定 checkpoint 随机抽取一张测试集图片预测（不指定默认 ResNet-18）
python inference.py --random checkpoints/baseline_resnet34/best.pth

# 全测试集评估
python inference.py --test checkpoints/baseline_resnet18/best.pth
```


## 可视化

使用 **swanlab** 记录训练曲线，训练开始会自动输出链接，可实时监控训练曲线。

也可在 `config.py` 中修改 `logger` 为 `wandb` 或 `none`。
