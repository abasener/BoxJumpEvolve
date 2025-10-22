import pygame
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (don't show plots)
import matplotlib.pyplot as plt

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 400  # Widened from 800 to 1400
GROUND_Y = SCREEN_HEIGHT - 50
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)

# Calculate base timeout: distance to goal / speed at 30 FPS, then add 10% buffer
START_X = 50
BASE_SPEED = 5  # pixels per frame (increased from 2 for faster runs)
BASE_TIME_FRAMES = (SCREEN_WIDTH - 100 - START_X) / BASE_SPEED  # frames to reach goal
TIMEOUT_FRAMES = int(BASE_TIME_FRAMES * 1.1)  # +10% buffer

# --- Level elements (spaced out more for wider screen) ---
STANDARD_GAP_WIDTH = 60  # Standard gap size for physics calculations
gaps = [(300, STANDARD_GAP_WIDTH), (900, STANDARD_GAP_WIDTH)]  # Spread out gaps (was 200, 500)
walls = [(600, 30)]  # Moved wall to middle (was 350)
platforms = [(1100, 40, 60)]  # Moved platform further (was 600)
GOAL_X = SCREEN_WIDTH - 100

# Calculate jump velocity needed to clear 1.5x standard gap
# Physics: with gravity = 0.5, horizontal speed = BASE_SPEED
# We want to jump across distance = 1.5 * STANDARD_GAP_WIDTH
# Time in air to travel that distance: t = (1.5 * STANDARD_GAP_WIDTH) / BASE_SPEED
# For parabolic motion, to return to same height: vy_initial = gravity * t / 2
# But we need to go up and down, so: vy_initial = gravity * t
GRAVITY = 0.5
target_jump_distance = 1.5 * STANDARD_GAP_WIDTH
time_in_air = target_jump_distance / BASE_SPEED
JUMP_VELOCITY = -(GRAVITY * time_in_air)  # Negative because up is negative

# Removed: Auto-place platforms on top of walls (now walls handle their own collision)

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

def draw_generation_info(gen, alive_count, time_remaining):
    screen.blit(font.render(f"Generation: {gen}", True, (0, 0, 0)), (10, 10))
    screen.blit(font.render(f"Alive: {alive_count}", True, (0, 0, 0)), (10, 40))
    # Convert frames to seconds for display
    seconds_remaining = time_remaining / 30.0
    screen.blit(font.render(f"Time: {seconds_remaining:.1f}s", True, (0, 0, 0)), (10, 70))

# --- Neural Network ---
class NeuralNetwork:
    def __init__(self, input_size=14, hidden1_size=16, hidden2_size=12, output_size=2):
        # Layer 1: input -> hidden1
        self.w1 = np.random.randn(hidden1_size, input_size)
        self.b1 = np.random.randn(hidden1_size, 1)
        # Layer 2: hidden1 -> hidden2
        self.w2 = np.random.randn(hidden2_size, hidden1_size)
        self.b2 = np.random.randn(hidden2_size, 1)
        # Layer 3: hidden2 -> output
        self.w3 = np.random.randn(output_size, hidden2_size)
        self.b3 = np.random.randn(output_size, 1)

    def forward(self, x):
        # First hidden layer
        z1 = 1 / (1 + np.exp(-np.clip(self.w1 @ x + self.b1, -50, 50)))
        # Second hidden layer
        z2 = 1 / (1 + np.exp(-np.clip(self.w2 @ z1 + self.b2, -50, 50)))
        # Output layer (2 outputs: jump decision, memory)
        z3 = 1 / (1 + np.exp(-np.clip(self.w3 @ z2 + self.b3, -50, 50)))
        return z3

    def clone(self):
        clone = NeuralNetwork(self.w1.shape[1], self.w1.shape[0], self.w2.shape[0], self.w3.shape[0])
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        clone.w3 = self.w3.copy()
        clone.b3 = self.b3.copy()
        return clone

# --- Agent ---
class Agent:
    def __init__(self, brain=None):
        self.nn = brain or NeuralNetwork()
        self.reset()

    def reset(self):
        self.x, self.y = START_X, GROUND_Y - 20
        self.vx, self.vy = BASE_SPEED, 0
        self.on_ground = True
        self.fitness = 0
        self.alive = True
        self.finished = False
        self.stuck_on_wall = False
        self.age = 0  # frames alive
        self.death_timer = 0  # frames since death for red flash effect
        self.memory = 0.5  # Memory state (starts neutral at 0.5)


    def get_inputs(self):
        def closest(elements):
            return sorted((e for e in elements if e[0] > self.x), key=lambda e: e[0])[0] if any(e[0] > self.x for e in elements) else None

        # Next gap (3 inputs)
        next_gap = closest(gaps)
        dx_gap = next_gap[0] - self.x if next_gap else 1000
        w_gap = next_gap[1] if next_gap else STANDARD_GAP_WIDTH
        y_gap_top = GROUND_Y if next_gap else GROUND_Y

        # Next wall (4 inputs)
        next_wall = closest(walls)
        dx_wall = next_wall[0] - self.x if next_wall else 1000
        w_wall = next_wall[1] if next_wall else 30
        y_wall_bottom = GROUND_Y - 40 if next_wall else GROUND_Y - 40
        y_wall_top = GROUND_Y if next_wall else GROUND_Y

        # Next platform (3 inputs)
        next_platform = closest(platforms)
        dx_platform = next_platform[0] - self.x if next_platform else 1000
        w_platform = next_platform[2] if next_platform else 60
        y_platform = GROUND_Y - next_platform[1] if next_platform else GROUND_Y

        # Agent state (3 inputs)
        agent_y = self.y
        agent_vy = self.vy
        agent_on_ground = 1.0 if self.on_ground else 0.0

        # Memory (1 input)
        memory_input = self.memory

        return np.array([
            [dx_gap],
            [w_gap],
            [y_gap_top],
            [dx_wall],
            [w_wall],
            [y_wall_bottom],
            [y_wall_top],
            [dx_platform],
            [w_platform],
            [y_platform],
            [agent_y],
            [agent_vy],
            [agent_on_ground],
            [memory_input],
        ])

    def update(self):
        if not self.alive:
            # Increment death timer for red flash effect
            if self.death_timer < 30:  # Flash for 1 second (30 frames)
                self.death_timer += 1
            return

        # Increment age and check timeout
        self.age += 1
        if self.age > TIMEOUT_FRAMES:
            self.alive = False
            return

        if self.x + 20 >= GOAL_X:
            self.alive = False
            self.finished = True
            return

        inputs = self.get_inputs()
        outputs = self.nn.forward(inputs)
        jump_prob = outputs[0][0]
        memory_output = outputs[1][0]

        # Update memory for next frame
        self.memory = memory_output

        if jump_prob > 0.5 and self.on_ground:
            self.vy = JUMP_VELOCITY  # Use calculated jump velocity
            self.on_ground = False

        # Check for wall collision BEFORE moving (horizontal check)
        self.stuck_on_wall = False
        agent_rect_future = (self.x + self.vx, self.y, 20, 20)
        for wx, ww in walls:
            wall_rect = (wx, GROUND_Y - 40, ww, 40)
            # Check if agent overlaps with wall horizontally and vertically
            if (agent_rect_future[0] < wall_rect[0] + wall_rect[2] and
                agent_rect_future[0] + agent_rect_future[2] > wall_rect[0] and
                agent_rect_future[1] < wall_rect[1] + wall_rect[3] and
                agent_rect_future[1] + agent_rect_future[3] > wall_rect[1]):
                self.stuck_on_wall = True
                break

        # Only move horizontally if not stuck on wall
        if not self.stuck_on_wall:
            self.x += self.vx

        self.vy += GRAVITY  # Use consistent gravity constant
        self.y += self.vy

        # Determine correct ground level (platform or floor)
        ground_level = GROUND_Y
        for px, ph, pw in platforms:
            if px <= self.x <= px + pw and GROUND_Y - ph < self.y + 20 < GROUND_Y:
                ground_level = GROUND_Y - ph

        # Check if agent should land on top of a wall (like a platform)
        for wx, ww in walls:
            wall_top_y = GROUND_Y - 40
            # If agent is falling onto the wall from above
            if (wx <= self.x + 20 and self.x <= wx + ww and
                wall_top_y < self.y + 20 <= wall_top_y + 5 and self.vy >= 0):
                ground_level = wall_top_y

        if self.y >= ground_level - 20:
            self.y = ground_level - 20
            self.vy = 0
            self.on_ground = True

        # Prevent clipping through bottom of walls (push agent down if inside wall)
        for wx, ww in walls:
            wall_top_y = GROUND_Y - 40
            wall_bottom_y = GROUND_Y
            # If agent is inside the wall vertically and horizontally
            if (wx < self.x + 20 and self.x < wx + ww and
                wall_top_y < self.y < wall_bottom_y):
                # Push agent below the wall
                if self.y < wall_top_y + 20:  # If more in the top half, push up
                    self.y = wall_top_y - 20
                    self.vy = 0
                    self.on_ground = True

        # Death by gap
        for gx, gw in gaps:
            if gx < self.x < gx + gw and self.y + 20 >= GROUND_Y:
                self.alive = False

        # Fitness calculation: distance traveled + huge bonus for finishing
        self.fitness = self.x
        if self.finished:
            self.fitness += 1000  # Big bonus for reaching the goal

    def draw(self):
        # Show red flash when dead (convert blue shade to red shade)
        if not self.alive and self.death_timer < 30:
            # Convert blue color (100, varying, 255) to red (255, varying, 100)
            # Keep the same intensity variation but swap red/blue channels
            blue_intensity = self.color[1]  # The varying middle channel
            red_color = (255, blue_intensity, 100)
            pygame.draw.rect(screen, red_color, (self.x, self.y, 20, 20))
        else:
            pygame.draw.rect(screen, self.color, (self.x, self.y, 20, 20))

# --- Evolution ---
def evolve(population, retain=0.2, mutate_chance=0.9, base_mutation=0.8, fresh_blood_rate=0.1):
    population.sort(key=lambda a: a.fitness, reverse=True)
    survivors = population[:int(len(population) * retain)]

    # ELITISM: Always keep the best agent unchanged
    children = [Agent(brain=population[0].nn.clone())]

    # Add some completely random agents for diversity (fresh blood)
    num_fresh = int(len(population) * fresh_blood_rate)
    for _ in range(num_fresh):
        children.append(Agent())  # Random new agent

    # Fitness-proportional selection: better agents more likely to be selected
    survivor_fitnesses = [agent.fitness for agent in survivors]
    fitness_sum = sum(survivor_fitnesses)

    # Handle edge case where all fitnesses are 0
    if fitness_sum == 0:
        selection_weights = [1.0 / len(survivors)] * len(survivors)
    else:
        selection_weights = [f / fitness_sum for f in survivor_fitnesses]

    while len(children) < len(population):
        # Select parent based on fitness (better = more likely)
        parent = random.choices(survivors, weights=selection_weights, k=1)[0]
        clone = parent.nn.clone()
        # Use actual distance (x position) for mutation calculation, not fitness with bonus
        score_ratio = parent.x / GOAL_X
        score_ratio = max(0.01, min(score_ratio, 1.0))
        mutation_strength = base_mutation * (1.5 - score_ratio)

        if random.random() < mutate_chance:
            for param in [clone.w1, clone.b1, clone.w2, clone.b2, clone.w3, clone.b3]:
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

        # Calculate time remaining (based on oldest agent's age)
        if alive_agents:
            max_age = max(agent.age for agent in alive_agents)
            time_remaining = max(0, TIMEOUT_FRAMES - max_age)
        else:
            time_remaining = 0

        draw_generation_info(generation_number, len(alive_agents), time_remaining)

        if not alive_agents or ticks > 1000:
            break

        for agent in agents:
            agent.update()
            # Draw all agents (alive or recently dead for red flash)
            if agent.alive or agent.death_timer < 30:
                agent.draw()

        pygame.display.flip()
        ticks += 1

    return agents

# --- Plotting Function ---
def save_fitness_plot(generations, avg_distances, max_distances, survival_counts):
    """Save a plot of average and max distances over generations."""
    plt.figure(figsize=(10, 6))

    # Add some padding to y-axis (10% above max value)
    y_max = GOAL_X * 1.1

    plt.plot(generations, avg_distances, 'b-o', label='Average Distance', linewidth=2)
    plt.plot(generations, max_distances, 'r-o', label='Max Distance', linewidth=2)
    plt.plot(generations, survival_counts, 'g-s', label='Survival Rate (scaled)', linewidth=2, alpha=0.7)

    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('Distance', fontsize=12)
    plt.title('Evolution Progress: Distance Traveled per Generation', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)

    # Set y-axis limits with padding
    plt.ylim(0, y_max)

    # Set x-axis to start at 0
    plt.xlim(0, max(generations) + 1)

    plt.tight_layout()
    plt.savefig('evolution_progress.png', dpi=150)
    plt.close()
    print("Plot saved as: evolution_progress.png")

# --- Main ---
NUM_GENERATIONS = 30
POP_SIZE = 21

# Track fitness over generations
generation_numbers = []
avg_distances_history = []
max_distances_history = []
survival_count_history = []  # Track number of finishers

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

    # Track data for plotting (use actual distance, not fitness with bonus)
    generation_numbers.append(gen)
    # Remove the bonus from finished agents for plotting purposes
    actual_distances = [a.x for a in agents]
    avg_distances_history.append(sum(actual_distances) / len(actual_distances))
    max_distances_history.append(max(actual_distances))
    # Map survival count to distance scale: 0 survivors = 0, all survivors = GOAL_X
    survival_count_history.append((num_finished / len(agents)) * GOAL_X)

    agents = evolve(agents)
    for agent in agents:
        agent.reset()
    agents = run_generation(POP_SIZE, generation_number=gen)

# Save the plot after all generations complete
save_fitness_plot(generation_numbers, avg_distances_history, max_distances_history, survival_count_history)

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
