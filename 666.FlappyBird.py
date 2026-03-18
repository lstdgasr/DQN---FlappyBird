import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import sys
import os
from torch.optim.lr_scheduler import StepLR

# -------------------------- 1. 全局参数配置（GPU优化版） --------------------------
# 游戏参数
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BIRD_WIDTH = 40
BIRD_HEIGHT = 30
BIRD_JUMP_VEL = -10
BIRD_GRAVITY = 0.5
PIPE_WIDTH = 60
PIPE_GAP = 180
PIPE_VEL = 5

# 训练模型参数
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 10000
GAMMA = 0.99
BATCH_SIZE = 96  # 显存足够
MEMORY_CAPACITY = 20000
LR = 1e-3
TARGET_UPDATE = 10
NUM_EPISODES = 20000

# 随机种子
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# GPU配置（自动检测+验证）
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    device = torch.device("cuda:0")
    print(f" 使用GPU训练: {torch.cuda.get_device_name(0)}")
    print(f" CUDA版本: {torch.version.cuda}")
else:
    device = torch.device("cpu")
    print(" 未检测到GPU，使用CPU训练（速度较慢）")

# -------------------------- 2. DQN模型（GPU+混合精度适配） --------------------------    PS。非CNN版本

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)     # dim少了欠拟合，多了过拟合，小游戏（64，128，256，512）选个128不大不小刚刚好
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)

# -------------------------- 3. 游戏环境（GPU） --------------------------   说实话小游戏用处不大，还是纯吃CPU
class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("FlappyBird RL ")
        self.clock = pygame.time.Clock()
        try:
            self.font = pygame.font.Font(pygame.font.get_default_font(), 40)
        except:
            self.font = None
        self.reset()

    def reset(self):
        self.bird_x = SCREEN_WIDTH // 4             # set初始位置
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_vel = 0                           # 小鸟垂直速度归零（初始静止状态）
        self.pipes = []                             # 初始no管道
        self._generate_pipe(SCREEN_WIDTH)
        self._generate_pipe(SCREEN_WIDTH + 200)     # 在第一个管道右侧200像素处生成第二个管道
        self.score = 0
        self.done = False
        return self._get_state()

    def _generate_pipe(self, x_pos):
        pipe_y = random.randint(150, SCREEN_HEIGHT - 150)
        upper_pipe = {
            "x": x_pos,
            "y": 0,
            "width": PIPE_WIDTH,
            "height": pipe_y - PIPE_GAP // 2
        }
        lower_pipe = {
            "x": x_pos,
            "y": pipe_y + PIPE_GAP // 2,
            "width": PIPE_WIDTH,
            "height": SCREEN_HEIGHT - (pipe_y + PIPE_GAP // 2)
        }
        self.pipes.append((upper_pipe, lower_pipe))

    def _get_state(self):
        closest_pipe = None             # 1. 找到离小鸟最近的管道
        min_dist = float("inf")
        for pipe_pair in self.pipes:
            pipe_x = pipe_pair[0]["x"]
            dist = pipe_x - (self.bird_x + BIRD_WIDTH)
            if dist >= -PIPE_WIDTH and dist < min_dist:
                min_dist = dist
                closest_pipe = pipe_pair
        if closest_pipe is None:
            closest_pipe = self.pipes[-1]

        # 2. 特征归一化（把不同尺度的特征缩放到统一范围）
        bird_y_norm = self.bird_y / SCREEN_HEIGHT           # 小鸟y坐标（0-1）
        bird_vel_norm = self.bird_vel / 20                  # 小鸟垂直速度（约-0.5~0.5）
        pipe_x_norm = min_dist / SCREEN_WIDTH               # 最近管道的水平距离（0-1）
        upper_pipe_bottom = closest_pipe[0]["height"]       # 上管道底部y坐标
        lower_pipe_top = closest_pipe[1]["y"]               # 下管道顶部y坐标
        dist_to_upper = (self.bird_y - upper_pipe_bottom) / SCREEN_HEIGHT  # 小鸟到上管道底部的垂直距离（归一化）
        dist_to_lower = (lower_pipe_top - self.bird_y) / SCREEN_HEIGHT   # 小鸟到下管道顶部的垂直距离（归一化）

        return np.array([
            bird_y_norm, bird_vel_norm, pipe_x_norm,
            dist_to_upper, dist_to_lower
        ], dtype=np.float32)    # float32适配PyTorch张量

    def step(self, action):
        if action == 1:
            self.bird_vel = BIRD_JUMP_VEL

        self.bird_vel += BIRD_GRAVITY
        self.bird_y += self.bird_vel

        for pipe_pair in self.pipes:
            pipe_pair[0]["x"] -= PIPE_VEL
            pipe_pair[1]["x"] -= PIPE_VEL

        while self.pipes and self.pipes[0][0]["x"] < -PIPE_WIDTH:
            self.pipes.pop(0)
        if len(self.pipes) < 2:
            last_pipe_x = self.pipes[-1][0]["x"]
            self._generate_pipe(last_pipe_x + 200)

        reward = 0.1
        self.done = False

        if self.bird_y < 0 or self.bird_y > SCREEN_HEIGHT - BIRD_HEIGHT:
            self.done = True
            reward = -15
        else:
            bird_rect = pygame.Rect(self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT)
            for pipe_pair in self.pipes:
                upper_rect = pygame.Rect(pipe_pair[0]["x"], pipe_pair[0]["y"], pipe_pair[0]["width"], pipe_pair[0]["height"])
                lower_rect = pygame.Rect(pipe_pair[1]["x"], pipe_pair[1]["y"], pipe_pair[1]["width"], pipe_pair[1]["height"])
                if bird_rect.colliderect(upper_rect) or bird_rect.colliderect(lower_rect):
                    self.done = True
                    reward = -15

        for pipe_pair in self.pipes:
            pipe_x = pipe_pair[0]["x"]
            if pipe_x + PIPE_WIDTH < self.bird_x and not pipe_pair[0].get("scored", False):
                pipe_pair[0]["scored"] = True
                self.score += 1
                reward = 20

        if not self.done:
            closest_pipe = self.pipes[0] if self.pipes else self.pipes[-1]
            pipe_center_y = (closest_pipe[0]["height"] + closest_pipe[1]["y"]) / 2
            bird_center_y = self.bird_y + BIRD_HEIGHT / 2
            vertical_diff = abs(bird_center_y - pipe_center_y)
            if vertical_diff < 50:
                reward -= 0.05

        return self._get_state(), reward, self.done

    def render(self):
        self.screen.fill((135, 206, 235))
        pygame.draw.rect(self.screen, (255, 255, 0), (self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))
        for pipe_pair in self.pipes:
            pygame.draw.rect(self.screen, (34, 139, 34), (pipe_pair[0]["x"], pipe_pair[0]["y"], pipe_pair[0]["width"], pipe_pair[0]["height"]))
            pygame.draw.rect(self.screen, (34, 139, 34), (pipe_pair[1]["x"], pipe_pair[1]["y"], pipe_pair[1]["width"], pipe_pair[1]["height"]))
        if self.font:
            score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))
            self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(0)

# -------------------------- 4. 训练逻辑（修复所有警告） --------------------------
def train():
    env = FlappyBirdEnv()
    input_dim = 5
    output_dim = 2

    # 模型初始化并移到GPU
    policy_net = DQN(input_dim, output_dim).to(device)  # 策略网络（实时更新，用于选动作）
    target_net = DQN(input_dim, output_dim).to(device)  # 目标网络（固定更新，用于计算目标Q值）
    target_net.load_state_dict(policy_net.state_dict())  # 初始时目标网络复制策略网络参数
    target_net.eval()                                   # 目标网络仅用于推理，不训练

    # 验证模型设备
    print(f" 模型设备: {next(policy_net.parameters()).device}")

    # 优化器+学习率调度
    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)    # 每5000减半LR

    scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else torch.amp.GradScaler('cpu')

    criterion = nn.MSELoss()

    memory = deque(maxlen=MEMORY_CAPACITY)  # 经验池（最多MEMORY_CAPACITY = 20000条）
    steps_done = 0

    # 训练监控
    best_score = 0
    score_history = deque(maxlen=100)

    print("\n 开始训练（20000轮）...")
    print("------------------------")

    for episode in range(NUM_EPISODES):
        state = env.reset()
        total_reward = 0
        episode_optimizer_steps = 0   # 记录本轮是否更新过网络（用于学习率调度）

        while True:
            # 处理退出事件
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # ε-贪心选动作 （  ε随step衰减：前期高ε（多探索），后期低ε（多利用））
            eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
            steps_done += 1

            if random.random() > eps_threshold:
                with torch.no_grad():
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    action = policy_net(state_tensor).argmax().item()
            else:
                action = random.choice([0, 1])

            # 执行动作
            next_state, reward, done = env.step(action)
            total_reward += reward
            memory.append((state, action, reward, next_state, done))
            state = next_state

            # 训练时注释render
            # env.render()

            # 经验回放（修复混合精度API + 调度器顺序）
            if len(memory) > BATCH_SIZE:
                batch = random.sample(memory, BATCH_SIZE)
                states = torch.tensor(np.array([x[0] for x in batch]), dtype=torch.float32).to(device)
                actions = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.long).to(device)
                rewards = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).to(device)
                next_states = torch.tensor(np.array([x[3] for x in batch]), dtype=torch.float32).to(device)
                dones = torch.tensor(np.array([x[4] for x in batch]), dtype=torch.float32).to(device)

                # PyTorch 2.0+ autocast API
                with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
                    with torch.no_grad():
                        next_q_values = target_net(next_states).max(1)[0]
                        target_q_values = rewards + (1 - dones) * GAMMA * next_q_values
                    loss = criterion(q_values, target_q_values)

                # 反向传播
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                episode_optimizer_steps += 1  # 标记本轮有优化

            if done:
                break

        #  修复：学习率调度器调用顺序（仅当本轮有优化时调用）
        if episode_optimizer_steps > 0:
            scheduler.step()

        # 记录得分
        score_history.append(env.score)
        if env.score > best_score:
            best_score = env.score
            torch.save(policy_net.state_dict(), "flappy_bird_dqn_best.pth")

        # 打印日志
        if (episode + 1) % 100 == 0:
            avg_score = np.mean(score_history)
            print(f"Episode {episode+1}/{NUM_EPISODES} | 平均得分: {avg_score:.1f} | 最佳得分: {best_score} | ε: {eps_threshold:.3f} | LR: {scheduler.get_last_lr()[0]:.6f}")
        elif (episode + 1) % 10 == 0:
            print(f"Episode {episode+1} | 得分: {env.score} | ε: {eps_threshold:.3f}")

        # 更新目标网络
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 早停
        if len(score_history) == 100 and np.mean(score_history) >= 20:
            print(f"\n 训练提前收敛。最近100轮平均得分: {np.mean(score_history):.1f}")
            break

    # 保存最终模型
    torch.save(policy_net.state_dict(), "flappy_bird_dqn_final.pth")
    print("\n 训练完成！")
    print(f" 最佳得分: {best_score}")
    print(f" 模型保存路径: flappy_bird_dqn_best.pth")
    pygame.quit()

# -------------------------- 5. 运行入口 --------------------------
if __name__ == "__main__":
    train()