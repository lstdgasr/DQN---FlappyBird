import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import sys
import os

# -------------------------- 1. 基础配置（和训练时保持一致） --------------------------
# 游戏参数（必须和训练时完全相同）
SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
BIRD_WIDTH = 40
BIRD_HEIGHT = 30
BIRD_JUMP_VEL = -10
BIRD_GRAVITY = 0.5
PIPE_WIDTH = 60
PIPE_GAP = 180
PIPE_VEL = 5

# 自动检测设备（GPU/CPU）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# -------------------------- 2. DQN模型定义（和训练时完全相同） --------------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.ln1 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 128)
        self.ln2 = nn.LayerNorm(128)
        self.fc3 = nn.Linear(128, output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.ln1(self.fc1(x)))
        x = self.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# -------------------------- 3. 游戏环境（和训练时完全相同） --------------------------
class FlappyBirdEnv:
    def __init__(self):
        pygame.init()
        pygame.font.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("FlappyBird RL - 模型测试-Final")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(pygame.font.get_default_font(), 40)
        self.reset()

    def reset(self):
        self.bird_x = SCREEN_WIDTH // 4
        self.bird_y = SCREEN_HEIGHT // 2
        self.bird_vel = 0
        self.pipes = []
        self._generate_pipe(SCREEN_WIDTH)
        self._generate_pipe(SCREEN_WIDTH + 200)
        self.score = 0
        self.done = False
        return self._get_state()

    def _generate_pipe(self, x_pos):
        pipe_y = random.randint(150, SCREEN_HEIGHT - 150)
        upper_pipe = {"x": x_pos, "y": 0, "width": PIPE_WIDTH, "height": pipe_y - PIPE_GAP // 2}
        lower_pipe = {"x": x_pos, "y": pipe_y + PIPE_GAP // 2, "width": PIPE_WIDTH,
                      "height": SCREEN_HEIGHT - (pipe_y + PIPE_GAP // 2)}
        self.pipes.append((upper_pipe, lower_pipe))

    def _get_state(self):
        closest_pipe = None
        min_dist = float("inf")
        for pipe_pair in self.pipes:
            pipe_x = pipe_pair[0]["x"]
            dist = pipe_x - (self.bird_x + BIRD_WIDTH)
            if dist >= -PIPE_WIDTH and dist < min_dist:
                min_dist = dist
                closest_pipe = pipe_pair
        if closest_pipe is None:
            closest_pipe = self.pipes[-1]

        bird_y_norm = self.bird_y / SCREEN_HEIGHT
        bird_vel_norm = self.bird_vel / 20
        pipe_x_norm = min_dist / SCREEN_WIDTH
        upper_pipe_bottom = closest_pipe[0]["height"]
        lower_pipe_top = closest_pipe[1]["y"]
        dist_to_upper = (self.bird_y - upper_pipe_bottom) / SCREEN_HEIGHT
        dist_to_lower = (lower_pipe_top - self.bird_y) / SCREEN_HEIGHT

        return np.array([bird_y_norm, bird_vel_norm, pipe_x_norm, dist_to_upper, dist_to_lower], dtype=np.float32)

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

        self.done = False
        if self.bird_y < 0 or self.bird_y > SCREEN_HEIGHT - BIRD_HEIGHT:
            self.done = True
        else:
            bird_rect = pygame.Rect(self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT)
            for pipe_pair in self.pipes:
                upper_rect = pygame.Rect(pipe_pair[0]["x"], pipe_pair[0]["y"], pipe_pair[0]["width"],
                                         pipe_pair[0]["height"])
                lower_rect = pygame.Rect(pipe_pair[1]["x"], pipe_pair[1]["y"], pipe_pair[1]["width"],
                                         pipe_pair[1]["height"])
                if bird_rect.colliderect(upper_rect) or bird_rect.colliderect(lower_rect):
                    self.done = True

        for pipe_pair in self.pipes:
            pipe_x = pipe_pair[0]["x"]
            if pipe_x + PIPE_WIDTH < self.bird_x and not pipe_pair[0].get("scored", False):
                pipe_pair[0]["scored"] = True
                self.score += 1

        return self._get_state(), 0.1, self.done

    def render(self):
        self.screen.fill((135, 206, 235))  # 天空蓝背景
        pygame.draw.rect(self.screen, (255, 255, 0), (self.bird_x, self.bird_y, BIRD_WIDTH, BIRD_HEIGHT))  # 黄色小鸟
        for pipe_pair in self.pipes:
            pygame.draw.rect(self.screen, (34, 139, 34), (
            pipe_pair[0]["x"], pipe_pair[0]["y"], pipe_pair[0]["width"], pipe_pair[0]["height"]))  # 绿色管道
            pygame.draw.rect(self.screen, (34, 139, 34),
                             (pipe_pair[1]["x"], pipe_pair[1]["y"], pipe_pair[1]["width"], pipe_pair[1]["height"]))
        score_text = self.font.render(f"Score: {self.score}", True, (0, 0, 0))  # 黑色分数
        self.screen.blit(score_text, (10, 10))
        pygame.display.flip()
        self.clock.tick(60)  # 60帧流畅展示


# -------------------------- 4. 加载模型并运行 --------------------------
def run_trained_model(model_path="flappy_bird_dqn_best.pth"):
    """加载训练好的模型运行"""
    # 检查模型文件
    if not os.path.exists(model_path):
        print(f" 未找到模型文件: {model_path}")
        print(f"请确认文件是否在当前目录，或修改model_path参数为正确的模型路径（比如flappy_bird_dqn_final.pth）")
        return

    # 初始化环境
    env = FlappyBirdEnv()

    # 初始化并加载模型
    model = DQN(input_dim=5, output_dim=2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()  # 设置为评估模式，禁用训练相关的层（如Dropout）

    print(f" 成功加载模型: {model_path}")
    print(f"运行设备: {device}")
    print(" 模型运行中...（按ESC或关闭窗口退出）")

    try:
        while True:
            state = env.reset()
            while not env.done:
                # 处理退出事件
                for event in pygame.event.get():
                    if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                        pygame.quit()
                        sys.exit()

                # 模型预测动作（纯贪心策略，无随机）
                with torch.no_grad():  # 禁用梯度计算，提升速度
                    state_tensor = torch.from_numpy(state).unsqueeze(0).to(device)
                    action = model(state_tensor).argmax().item()

                # 执行动作
                state, _, _ = env.step(action)

                # 渲染画面（核心：展示游戏效果）
                env.render()

            print(f" 本局得分: {env.score} | 自动重新开始...")

    except KeyboardInterrupt:
        pygame.quit()
        print("\n 测试结束")


# -------------------------- 5. 直接运行 --------------------------
if __name__ == "__main__":
    # 默认加载最佳模型，如果你想加载最终模型，修改为 "flappy_bird_dqn_final.pth"
    run_trained_model(model_path="flappy_bird_dqn_best.pth")