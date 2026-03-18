# DQN---FlappyBird
DQN经典FlappyBird

通过自生成
Flappy Bird - CNN+DQN 强化学习自动游戏
基于 PyGame 实现 Flappy Bird 游戏环境，结合 **CNN+DQN（深度 Q 网络）** 强化学习算法，让 AI 自动学习玩小鸟游戏，支持 GPU 自动加速训练。
项目简介
本项目实现端到端强化学习训练流程：
用 PyGame 搭建轻量化 Flappy Bird 原生游戏环境
OpenCV 对游戏画面预处理（灰度化 / 缩放 / 归一化）
构建 CNN 卷积神经网络提取视觉特征
DQN 核心算法（经验回放 + 目标网络）训练 AI 智能体
自动保存训练模型，支持 GPU/CPU 无缝切换
环境依赖
执行以下命令安装所有依赖库：
bash
运行
pip install pygame numpy opencv-python torch
基础环境要求
Python 3.8+
PyTorch 1.8+（支持 CUDA）
OpenCV 4.0+
PyGame 2.0+
核心模块划分
代码逻辑清晰，分为 5 大核心模块：
FlappyBird 游戏环境：游戏重置、动作执行、碰撞检测、画面渲染
图像预处理：游戏帧转 84×84 灰度图，堆叠 4 帧构建网络输入
CNN-DQN 网络模型：3 层卷积层 + 2 层全连接层，提取游戏视觉特征
DQN 智能体：ε- 贪心策略、经验回放、网络训练、目标网络同步
主训练循环：控制训练流程、日志打印、模型自动保存
可配置超参数
内置优化超参数，适配 NVIDIA 3050Ti 等中端显卡，直接在代码中修改：
python
运行
# 设备配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 游戏窗口参数
SCREEN_WIDTH = 288
SCREEN_HEIGHT = 512
ACTION_SPACE = 2  # 0=不跳 1=跳
# DQN核心训练参数
LEARNING_RATE = 1e-4
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY = 0.99995
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_STEP = 1000
STACK_FRAMES = 4
IMAGE_SIZE = 84
快速运行
安装所有依赖库
直接运行代码文件：
bash
运行
python 文件名.py
自动弹出游戏窗口，AI 开始自主训练
每 100 回合自动保存模型：flappy_bird_cnn_dqn.pth
训练机制说明
1. 动作选择策略
ε- 贪心算法：初始高概率随机探索（学习玩法），探索率逐步衰减，最终完全依赖 AI 决策。
2. 奖励函数设计
小鸟存活：+0.1 奖励
成功穿过管道：+1 奖励
碰撞 / 飞出边界：-1 惩罚
3. 训练稳定性优化
经验回放池：打破数据相关性，提升训练效率
双网络结构：策略网络 + 目标网络，防止训练震荡
固定步长更新目标网络，保证收敛性
训练日志输出
控制台实时打印训练核心信息：
plaintext
回合: XXX | 分数: XX | 总奖励: X.X | 探索率: X.XXX
回合：当前训练轮次
分数：小鸟本局得分
总奖励：本局累计奖励
探索率：AI 随机探索概率（逐步降低）
代码优化特性
自动检测 CUDA，优先使用 GPU 训练
修复 PyGame 窗口无响应问题（内置事件循环）
使用 PyGame 内置字体，解决系统字体缺失报错
优化图像预处理，降低网络计算量
适配显卡显存，批次大小 32 完美适配中端 GPU
注意事项
终止程序：直接关闭 PyGame 游戏窗口即可
模型复用：可通过torch.load()加载已保存的训练模型
训练速度：GPU 训练速度远快于 CPU，建议使用 NVIDIA 显卡
无卡顿运行：已修复 PyGame 渲染卡顿问题
