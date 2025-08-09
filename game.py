
# --- Snake Game Environment for DQN ---
import pygame
import random
import numpy as np

pygame.init()

WIDTH, HEIGHT = 640, 480
BLOCK_SIZE = 20

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (200, 0, 0)
GREEN = (0, 200, 0)

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)


class SnakeGameEnv:
    def __init__(self, speed=10):
        self.display = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('Snake DQN')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('arial', 25)
        self.speed = speed
        self.reset()

    def reset(self):
        self.direction = RIGHT
        self.snake = [(WIDTH // 2, HEIGHT // 2),
                      (WIDTH // 2 - BLOCK_SIZE, HEIGHT // 2)]
        self.grow = False
        self.score = 0
        self.food = self._place_food()
        self.frame = 0
        return self.get_state()

    def _place_food(self):
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) //
                               BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) //
                               BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake:
                return (x, y)

    def step(self, action):
        # action: 0=straight, 1=right, 2=left
        self.frame += 1
        self._move(action)
        reward = 0
        done = False
        if self._is_collision(self.snake[0]) or self.frame > 100*len(self.snake):
            done = True
            reward = -10
            return self.get_state(), reward, done, self.score
        if self.snake[0] == self.food:
            self.grow = True
            self.score += 1
            reward = 10
            self.food = self._place_food()
        else:
            self.grow = False
        if not self.grow:
            self.snake.pop()
        self._update_ui()
        self.clock.tick(self.speed)
        return self.get_state(), reward, done, self.score

    def _move(self, action):
        directions = [RIGHT, DOWN, LEFT, UP]
        idx = directions.index(self.direction)
        if action == 0:  # straight
            new_dir = directions[idx]
        elif action == 1:  # right turn
            new_dir = directions[(idx + 1) % 4]
        else:  # left turn
            new_dir = directions[(idx - 1) % 4]
        self.direction = new_dir
        x, y = self.snake[0]
        dx, dy = self.direction
        new_head = (x + dx * BLOCK_SIZE, y + dy * BLOCK_SIZE)
        self.snake.insert(0, new_head)

    def _is_collision(self, pt):
        x, y = pt
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pos in self.snake:
            pygame.draw.rect(self.display, GREEN,
                             (*pos, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(self.display, RED,
                         (*self.food, BLOCK_SIZE, BLOCK_SIZE))
        score_text = self.font.render(f"Score: {self.score}", True, WHITE)
        self.display.blit(score_text, (10, 10))
        pygame.display.flip()

    def get_state(self):
        head = self.snake[0]
        point_l = (head[0] - BLOCK_SIZE, head[1])
        point_r = (head[0] + BLOCK_SIZE, head[1])
        point_u = (head[0], head[1] - BLOCK_SIZE)
        point_d = (head[0], head[1] + BLOCK_SIZE)
        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        state = [
            # Danger straight
            (dir_r and self._is_collision(point_r)) or
            (dir_l and self._is_collision(point_l)) or
            (dir_u and self._is_collision(point_u)) or
            (dir_d and self._is_collision(point_d)),

            # Danger right
            (dir_u and self._is_collision(point_r)) or
            (dir_d and self._is_collision(point_l)) or
            (dir_l and self._is_collision(point_u)) or
            (dir_r and self._is_collision(point_d)),

            # Danger left
            (dir_d and self._is_collision(point_r)) or
            (dir_u and self._is_collision(point_l)) or
            (dir_r and self._is_collision(point_u)) or
            (dir_l and self._is_collision(point_d)),

            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # Food location
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1],  # food down
        ]
        return np.array(state, dtype=int)
