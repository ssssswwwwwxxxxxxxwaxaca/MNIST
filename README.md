# MNIST 手写数字识别项目

## 项目简介

本项目基于 PyTorch 实现了对 MNIST 手写数字图片的深度学习识别，包含数据加载、模型训练、测试评估、结果可视化等完整流程。项目结构清晰，适合深度学习入门和实验。

---

## 目录结构

```
e:\MNIST
│
├─code/                # 主要代码文件及训练脚本
│   ├─train.py         # 训练主程序（可直接运行）
│   ├─train.ipynb      # Jupyter Notebook 版训练与分析
│   ├─train.png        # 训练过程损失/准确率曲线图
│   ├─result_.png      # 测试集预测结果可视化
│   └─data/            # MNIST原始数据
│       └─MNIST/
│           └─raw/     # 原始数据文件
│
├─model/               # 保存的模型参数
│   └─mycnn.pth        # 训练好的CNN模型
│
├─result/              # 结果可视化图片
│   ├─kernels.png      # 卷积核可视化
│   ├─output.png       # 其他输出结果
│   └─result.png       # 测试集预测结果
│
└─mnist.pth            # 训练好的模型参数（备份）
```

---

## 主要功能

- **数据加载**：自动下载并加载 MNIST 数据集
- **模型定义**：自定义卷积神经网络（CNN），支持 BatchNorm、Dropout
- **训练过程**：支持多轮训练，输出损失和准确率曲线
- **测试评估**：输出测试集准确率，支持单批次预测可视化
- **卷积核可视化**：展示模型第一层卷积核学习到的特征
- **模型保存与加载**：训练完成后自动保存模型参数，便于后续推理或复现

---

## 快速开始

1. **环境准备**

   推荐使用 Python 3.8+，安装依赖：

   ```
   pip install torch torchvision matplotlib scikit-learn pandas
   ```

2. **训练模型**

   运行训练脚本：

   ```
   python code/train.py
   ```

   或在 Jupyter Notebook 中运行 `train.ipynb`，可获得更丰富的可视化分析。

3. **查看结果**

   - 训练过程曲线见 `code/train.png`
   - 测试集预测结果见 `code/result_.png` 或 `result/result.png`
   - 卷积核可视化见 `result/kernels.png`
   - 模型参数保存在 `model/mycnn.pth` 或 `mnist.pth`

---

## 进阶功能

- 支持自定义网络结构和参数调整
- 可扩展至 FashionMNIST、CIFAR-10 等其他数据集
- 可添加混淆矩阵、错误样本分析等更高级可视化

---

## 参考

- [PyTorch 官方文档](https://pytorch.org/)
- [MNIST 数据集介绍](http://yann.lecun.com/exdb/mnist/)
- [深度学习入门教程](https://pytorch.org/tutorials/)

---

如有问题或建议，欢迎反馈。
