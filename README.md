# PRS-Net 复现

本项目是论文 **"PRS-Net: Planar Reflective Symmetry Detection Net for 3D Models"** 的一个非官方PyTorch实现。

该项目旨在复现论文的核心思想，包括其无监督学习框架、独特的损失函数以及验证逻辑。


## 环境设置

本项目在Python 3.13环境下运行。

1.  **克隆或下载项目代码**
    ```bash
    git clone https://github.com/JZLshen/PRS-Net-demo.git
    ```
2.  **安装依赖**
    在项目根目录下，使用 `pip` 安装所有必需的库：
    ```bash
    pip install -r requirements.txt
    ```

## 如何运行

直接运行主训练脚本即可启动一个使用示例模型（长方体）的完整训练和验证流程。

```bash
python train.py
```

## 文件说明

* **`config.py`**: 存储所有全局配置和超参数。
* **`model.py`**: 定义 `PRSNet` 神经网络的结构。
* **`loss.py`**: 实现核心的 `SymmetryLoss`，包含对称距离损失和正则化损失。
* **`utils.py`**: 包含数据预处理（体素化、点采样）和结果验证（根据误差和角度过滤）等辅助功能。
* **`train.py`**: 项目的启动文件，负责加载数据、初始化模型、执行训练循环并展示最终结果。