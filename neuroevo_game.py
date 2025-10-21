import pygame
import numpy as np
import random

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 400
GROUND_Y = SCREEN_HEIGHT - 50
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# --- Level elements ---
gaps = [(200, 60), (500, 60)]
walls = [(350, 30)]  # (x position, width)
platforms = [(600, 40, 60)]  # (x, height above ground, width)
GOAL_X = SCREEN_WIDTH - 100

# Auto-place platforms on top of walls
for wx, ww in walls:
    wall_top_y = 40
    platforms.append((wx, wall_top_y + 2, ww))  # tiny buffer for clean contact

# --- Drawing ---
def draw_level():
    screen.fill((255, 255, 255))
    pygame.draw.rect(screen, (0, 0, 0), (0, GROUND_Y, SCREEN_WIDTH, 50))
    for gx, gw in gaps:
        pygame.draw.rect(screen, (255, 255, 255), (gx, GROUND_Y, gw, 50))
    for wx, ww in walls:
        pygame.draw.rect(screen, (100, 100, 100), (wx, GROUND_Y - 40, ww, 40))
    for px, ph, pw in platforms:
        pygame.draw.rect(screen, (150, 75, 0), (px, GROUND_Y - ph, pw, 10))
    pygame.draw.line(screen, (0, 200, 0), (GOAL_X, 0), (GOAL_X, SCREEN_HEIGHT), 2)

def draw_generation_info(gen, alive_count):
    screen.blit(font.render(f"Generation: {gen}", True, (0, 0, 0)), (10, 10))
    screen.blit(font.render(f"Alive: {alive_count}", True, (0, 0, 0)), (10, 40))

# --- Neural Network ---
class NeuralNetwork:
    def __init__(self, input_size=7, hidden_size=6, output_size=1):
        self.w1 = np.random.randn(hidden_size, input_size)
        self.b1 = np.random.randn(hidden_size, 1)
        self.w2 = np.random.randn(output_size, hidden_size)
        self.b2 = np.random.randn(output_size, 1)

    def forward(self, x):
        z1 = 1 / (1 + np.exp(-np.clip(self.w1 @ x + self.b1, -50, 50)))
        z2 = 1 / (1 + np.exp(-np.clip(self.w2 @ z1 + self.b2, -50, 50)))
        return z2

    def clone(self):
        clone = NeuralNetwork(self.w1.shape[1], self.w1.shape[0], self.w2.shape[0])
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        return clone

# --- Agent ---
class Agent:
    def __init__(self, brain=None):
        self.nn = brain or NeuralNetwork()
        self.reset()

    def reset(self):
        self.x, self.y = 50, GROUND_Y - 20
        self.vx, self.vy = 2, 0
        self.on_ground = True
        self.fitness = 0
        self.alive = True
        self.finished = False


    def get_inputs(self):
        def closest(elements):
            return sorted((e for e in elements if e[0] > self.x), key=lambda e: e[0])[0] if any(e[0] > self.x for e in elements) else None

        # Next gap
        next_gap = closest(gaps)
        dx_gap = next_gap[0] - self.x if next_gap else 1000
        w_gap = next_gap[1] if next_gap else 50

        # Next wall
        next_wall = closest(walls)
        dx_wall = next_wall[0] - self.x if next_wall else 1000
        h_wall = 40 if next_wall else 40

        # Next platform
        next_platform = closest(platforms)
        dx_platform = next_platform[0] - self.x if next_platform else 1000
        h_platform = next_platform[1] if next_platform else 0

        return np.array([
            [dx_gap],
            [w_gap],
            [dx_wall],
            [h_wall],
            [dx_platform],
            [h_platform],
            [self.vy],
        ])

    def update(self):
        if not self.alive:
            return

        if self.x + 20 >= GOAL_X:
            self.alive = False
            self.finished = True
            return

        inputs = self.get_inputs()
        jump_prob = self.nn.forward(inputs)[0][0]
        if jump_prob > 0.5 and self.on_ground:
            self.vy = -10
            self.on_ground = False

        self.x += self.vx
        self.vy += 0.5
        self.y += self.vy

        # Determine correct ground level (platform or floor)
        ground_level = GROUND_Y
        for px, ph, pw in platforms:
            if px <= self.x <= px + pw and GROUND_Y - ph < self.y + 20 < GROUND_Y:
                ground_level = GROUND_Y - ph

        if self.y >= ground_level - 20:
            self.y = ground_level - 20
            self.vy = 0
            self.on_ground = True

        # Death by gap
        for gx, gw in gaps:
            if gx < self.x < gx + gw and self.y + 20 >= GROUND_Y:
                self.alive = False

        self.fitness = self.x

    def draw(self):
        pygame.draw.rect(screen, self.color, (self.x, self.y, 20, 20))

# --- Evolution ---
def evolve(population, retain=0.2, mutate_chance=0.9, base_mutation=0.5):
    population.sort(key=lambda a: a.fitness, reverse=True)
    survivors = population[:int(len(population) * retain)]
    children = []

    while len(children) < len(population):
        parent = random.choice(survivors)
        clone = parent.nn.clone()
        score_ratio = parent.fitness / GOAL_X
        score_ratio = max(0.01, min(score_ratio, 1.0))
        mutation_strength = base_mutation * (1.5 - score_ratio)

        if random.random() < mutate_chance:
            for param in [clone.w1, clone.b1, clone.w2, clone.b2]:
                param += np.random.randn(*param.shape) * mutation_strength

        children.append(Agent(brain=clone))

    return children

# --- Simulation Loop ---
def run_generation(pop_size=20, generation_number=0):
    agents = [Agent() for _ in range(pop_size)]
    for idx, agent in enumerate(agents):

        agent.color = (100, 100 + int(155 * idx / pop_size), 255)  # light to deeper blue
    running = True
    ticks = 0

    while running:
        clock.tick(30)
        draw_level()
        alive_agents = [a for a in agents if a.alive]
        draw_generation_info(generation_number, len(alive_agents))

        if not alive_agents or ticks > 1000:
            break

        for agent in agents:
            agent.update()
            if agent.alive:
                agent.draw()

        pygame.display.flip()
        ticks += 1

    return agents

# --- Main ---
NUM_GENERATIONS = 24
POP_SIZE = 21
agents = run_generation(POP_SIZE, generation_number=0)

for gen in range(1, NUM_GENERATIONS + 1):
    fitnesses = [a.fitness for a in agents]
    avg_fitness = sum(fitnesses) / len(fitnesses)
    max_fitness = max(fitnesses)
    num_finished = sum(1 for a in agents if getattr(a, 'finished', False))
    num_failed = len(agents) - num_finished

    print(f"Generation {gen}:")
    print(f"  Finished: {num_finished} / {len(agents)}")
    print(f"  Failed:   {num_failed} / {len(agents)}")
    print(f"  Avg distance: {avg_fitness:.1f}")
    print(f"  Max distance: {max_fitness:.1f}")

    agents = evolve(agents)
    for agent in agents:
        agent.reset()
    agents = run_generation(POP_SIZE, generation_number=gen)

# Final best agent
best_agent = max(agents, key=lambda a: a.fitness)
best_agent.reset()
running = True
while running and best_agent.alive:
    clock.tick(30)
    draw_level()
    best_agent.update()
    best_agent.draw()
    pygame.display.flip()
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
