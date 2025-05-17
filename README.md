# DQN Atari游戏实现

## 概述
本仓库包含一个使用PyTorch实现的深度Q网络(DQN)，用于玩Atari Breakout(打砖块)游戏。实现包含经验回放、目标网络更新和ε-贪婪探索策略。

## 环境要求
- Python 3.7+
- PyTorch >= 1.13.0
- torchvision >= 0.14.0
- gym == 0.26.1
- ale-py == 0.8.1
- opencv-python >= 4.5
- numpy == 1.23.5
- matplotlib >= 3.5
- tensorboard >= 2.10

## 安装步骤
1. 克隆本仓库:
   ```bash
   https://github.com/AlexLikka/DQN_Atari-Ai.git
   ```
   
2. 安装依赖包:
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法
1. 运行训练脚本:
   ```bash
   python DQN.py
   ```

2. 使用TensorBoard监控训练过程:
   ```bash
   tensorboard --logdir=runs
   ```

## 主要特性
- **深度Q网络架构**: 使用卷积层处理游戏画面帧
- **经验回放**: 存储并采样过去的经验用于训练
- **目标网络**: 使用独立网络提供稳定的Q值目标
- **ε-贪婪探索**: 平衡探索与利用
- **帧堆叠**: 使用连续4帧画面作为状态输入
- **Atari预处理**: 包含标准的Atari游戏预处理

## 超参数配置
- 批量大小: 32
- 折扣因子(γ): 0.99
- 初始ε: 1.0
- 最终ε: 0.01
- ε衰减率: 0.995
- 目标网络更新频率: 每10个episode
- 经验回放容量: 10,000
- 学习率: 0.00025
- 总训练episode数: 1,000

## 文件结构
- `DQN.py`: 主训练脚本
- `runs/`: 包含训练日志和模型检查点的目录

## 训练结果
脚本会自动保存:
- 每100个episode的模型检查点
- 训练完成后的最终模型
- TensorBoard格式的训练指标数据

## 注意事项
- 代码会自动检测并使用GPU(如果可用)，否则使用CPU
- 设置了随机种子以保证可复现性
- 可以通过设置脚本中的`RENDER = True`来启用游戏画面渲染
