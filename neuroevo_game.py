import pygame
import numpy as np
import random
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (don't show plots)
import matplotlib.pyplot as plt
from maps import load_map

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

# --- MAP SELECTION ---
# Using RANDOM maps with difficulty distribution
MAP_DIFFICULTY_MEAN = 5
MAP_SPREAD_LEFT = 4
MAP_SPREAD_RIGHT = 4

# Map will be loaded in main loop (regenerates every 3 generations)
# These will be set globally when map is loaded
gaps = []
walls = []
platforms = []
GOAL_X = SCREEN_WIDTH - 100

# For compatibility with agent input code
STANDARD_GAP_WIDTH = 60

# Set initial window title
pygame.display.set_caption(f"BoxJumpEvolve - Random Maps (mean={MAP_DIFFICULTY_MEAN})")

# Function to load a new random map
def load_new_map():
    """Load a new random map and update global obstacle lists."""
    global gaps, walls, platforms
    level_data = load_map('RANDOM', SCREEN_WIDTH,
                          difficulty_mean=MAP_DIFFICULTY_MEAN,
                          spread_left=MAP_SPREAD_LEFT,
                          spread_right=MAP_SPREAD_RIGHT)
    gaps = level_data['gaps']
    walls = level_data['walls']
    platforms = level_data['platforms']
    return level_data['difficulty']

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
    for wx, ww, wh in walls:
        pygame.draw.rect(screen, (100, 100, 100), (wx, GROUND_Y - wh, ww, wh))
    for px, ph, pw in platforms:
        pygame.draw.rect(screen, (150, 75, 0), (px, GROUND_Y - ph, pw, 10))
    pygame.draw.line(screen, (0, 200, 0), (GOAL_X, 0), (GOAL_X, SCREEN_HEIGHT), 2)

def draw_generation_info(gen, alive_count, time_remaining, map_difficulty):
    screen.blit(font.render(f"Generation: {gen}", True, (0, 0, 0)), (10, 10))
    screen.blit(font.render(f"Alive: {alive_count}", True, (0, 0, 0)), (10, 40))
    # Convert frames to seconds for display
    seconds_remaining = time_remaining / 30.0
    screen.blit(font.render(f"Time: {seconds_remaining:.1f}s", True, (0, 0, 0)), (10, 70))
    screen.blit(font.render(f"Map Diff: {map_difficulty}", True, (100, 100, 100)), (10, 100))

# --- Neural Network ---
class NeuralNetwork:
    def __init__(self, input_size=17, hidden1_size=24, hidden2_size=20, hidden3_size=16, output_size=2):
        # Layer 1: input -> hidden1
        self.w1 = np.random.randn(hidden1_size, input_size)
        self.b1 = np.random.randn(hidden1_size, 1)
        # Layer 2: hidden1 -> hidden2
        self.w2 = np.random.randn(hidden2_size, hidden1_size)
        self.b2 = np.random.randn(hidden2_size, 1)
        # Layer 3: hidden2 -> hidden3
        self.w3 = np.random.randn(hidden3_size, hidden2_size)
        self.b3 = np.random.randn(hidden3_size, 1)
        # Layer 4: hidden3 -> output
        self.w4 = np.random.randn(output_size, hidden3_size)
        self.b4 = np.random.randn(output_size, 1)

    def forward(self, x):
        # First hidden layer
        z1 = 1 / (1 + np.exp(-np.clip(self.w1 @ x + self.b1, -50, 50)))
        # Second hidden layer
        z2 = 1 / (1 + np.exp(-np.clip(self.w2 @ z1 + self.b2, -50, 50)))
        # Third hidden layer
        z3 = 1 / (1 + np.exp(-np.clip(self.w3 @ z2 + self.b3, -50, 50)))
        # Output layer (2 outputs: jump decision, memory)
        z4 = 1 / (1 + np.exp(-np.clip(self.w4 @ z3 + self.b4, -50, 50)))
        return z4

    def clone(self):
        clone = NeuralNetwork(self.w1.shape[1], self.w1.shape[0], self.w2.shape[0], self.w3.shape[0], self.w4.shape[0])
        clone.w1 = self.w1.copy()
        clone.b1 = self.b1.copy()
        clone.w2 = self.w2.copy()
        clone.b2 = self.b2.copy()
        clone.w3 = self.w3.copy()
        clone.b3 = self.b3.copy()
        clone.w4 = self.w4.copy()
        clone.b4 = self.b4.copy()
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
        h_wall = next_wall[2] if next_wall else 40
        y_wall_bottom = GROUND_Y - h_wall if next_wall else GROUND_Y - 40
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

        # NEW: Context inputs (3 inputs)
        # Height above ground (how elevated is the agent?)
        height_above_ground = GROUND_Y - (self.y + 20)  # +20 because y is top of agent

        # Am I on top of a wall? (boolean - helps identify "just cleared tall wall" state)
        on_wall = 0.0
        for wx, ww, wh in walls:
            wall_top_y = GROUND_Y - wh
            # Check if agent is standing on this wall
            if (wx <= self.x + 20 and self.x <= wx + ww and
                abs(self.y + 20 - wall_top_y) < 5 and self.on_ground):
                on_wall = 1.0
                break

        # Vertical distance from current position to next gap (will be negative if gap is below)
        dy_gap = y_gap_top - (self.y + 20)  # Distance from agent bottom to gap

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
            [height_above_ground],  # NEW
            [on_wall],              # NEW
            [dy_gap],               # NEW
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
        for wx, ww, wh in walls:
            wall_rect = (wx, GROUND_Y - wh, ww, wh)
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
        for wx, ww, wh in walls:
            wall_top_y = GROUND_Y - wh
            # If agent is falling onto the wall from above
            if (wx <= self.x + 20 and self.x <= wx + ww and
                wall_top_y < self.y + 20 <= wall_top_y + 5 and self.vy >= 0):
                ground_level = wall_top_y

        if self.y >= ground_level - 20:
            self.y = ground_level - 20
            self.vy = 0
            self.on_ground = True

        # Prevent clipping through bottom of walls (push agent down if inside wall)
        for wx, ww, wh in walls:
            wall_top_y = GROUND_Y - wh
            wall_bottom_y = GROUND_Y
            # If agent is inside the wall vertically and horizontally
            if (wx < self.x + 20 and self.x < wx + ww and
                wall_top_y < self.y < wall_bottom_y):
                # Push agent below the wall
                if self.y < wall_top_y + 20:  # If more in the top half, push up
                    self.y = wall_top_y - 20
                    self.vy = 0
                    self.on_ground = True

        # Death by gap - only die if 50% or more of body is in the gap
        # Agent is 20px wide (self.x to self.x + 20)
        # Check what percentage of the agent overlaps with each gap
        for gx, gw in gaps:
            if self.y + 20 >= GROUND_Y:  # Agent is at ground level
                # Calculate overlap between agent (self.x to self.x + 20) and gap (gx to gx + gw)
                overlap_start = max(self.x, gx)
                overlap_end = min(self.x + 20, gx + gw)
                overlap_width = max(0, overlap_end - overlap_start)

                # If 50% or more of agent's body (>=10px) is in the gap, they fall and die
                if overlap_width >= 10:
                    self.alive = False
                    break

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
def evolve(population, retain=0.4, mutate_chance=0.9, base_mutation=0.8, fresh_blood_rate=0.1):
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
def run_generation(pop_size=20, generation_number=0, map_difficulty=0):
    """Create random agents and run generation (used for generation 0 only)"""
    agents = [Agent() for _ in range(pop_size)]
    return run_generation_with_agents(agents, generation_number, map_difficulty)

def run_generation_with_agents(agents, generation_number=0, map_difficulty=0):
    """Run a generation with the provided agents (used for evolved populations)"""
    for idx, agent in enumerate(agents):
        agent.color = (100, 100 + int(155 * idx / len(agents)), 255)  # light to deeper blue
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

        draw_generation_info(generation_number, len(alive_agents), time_remaining, map_difficulty)

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

# --- Diversity Calculation ---
def calculate_diversity(population):
    """Calculate population diversity as average pairwise weight distance."""
    if len(population) < 2:
        return 0.0

    # Sample a subset for efficiency (compare 10 random pairs)
    num_samples = min(10, len(population) * (len(population) - 1) // 2)
    distances = []

    for _ in range(num_samples):
        # Pick two random agents
        i, j = random.sample(range(len(population)), 2)
        agent1, agent2 = population[i], population[j]

        # Calculate Euclidean distance between all weights
        dist = 0.0
        for p1, p2 in [(agent1.nn.w1, agent2.nn.w1), (agent1.nn.b1, agent2.nn.b1),
                       (agent1.nn.w2, agent2.nn.w2), (agent1.nn.b2, agent2.nn.b2),
                       (agent1.nn.w3, agent2.nn.w3), (agent1.nn.b3, agent2.nn.b3)]:
            dist += np.sum((p1 - p2) ** 2)
        distances.append(np.sqrt(dist))

    return np.mean(distances)

# --- Plotting Function ---
def save_fitness_plot(generations, avg_distances, max_distances, survival_counts, diversity_history, map_difficulties):
    """Save a plot with fitness progress and diversity metrics."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

    # Top plot: Evolution Progress
    y_max = GOAL_X * 1.1
    ax1.plot(generations, avg_distances, 'b-o', label='Average Distance', linewidth=2)
    ax1.plot(generations, max_distances, 'r-o', label='Max Distance', linewidth=2)
    ax1.plot(generations, survival_counts, 'g-s', label='Survival Rate (scaled)', linewidth=2, alpha=0.7)

    # Add map difficulty line (scaled to distance range for visibility)
    # Scale difficulty to fit within the plot (difficulty range varies, scale to 0-GOAL_X)
    if map_difficulties:
        max_diff = max(map_difficulties)
        min_diff = min(map_difficulties)
        if max_diff > min_diff:
            # Normalize to 0-1, then scale to 0-GOAL_X range
            scaled_difficulties = [(d - min_diff) / (max_diff - min_diff) * GOAL_X for d in map_difficulties]
        else:
            # All difficulties are the same
            scaled_difficulties = [GOAL_X / 2 for _ in map_difficulties]
        ax1.plot(generations, scaled_difficulties, 'orange', linestyle='--', marker='d',
                 label=f'Map Difficulty (scaled {min_diff}-{max_diff})', linewidth=2, alpha=0.8)

    ax1.set_ylabel('Distance', fontsize=12)
    ax1.set_title('Evolution Progress: Distance Traveled per Generation', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, y_max)

    # Bottom plot: Population Diversity
    ax2.plot(generations, diversity_history, 'm-o', label='Population Diversity', linewidth=2)
    ax2.set_xlabel('Generation', fontsize=12)
    ax2.set_ylabel('Diversity (Weight Distance)', fontsize=12)
    ax2.set_title('Population Diversity Over Generations', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, max(generations) + 1)

    plt.tight_layout()
    plt.savefig('evolution_progress.png', dpi=150)
    plt.close()
    print("Plot saved as: evolution_progress.png")

# --- Main ---
NUM_GENERATIONS = 60
POP_SIZE = 30

# Track fitness over generations
generation_numbers = []
avg_distances_history = []
max_distances_history = []
survival_count_history = []  # Track number of finishers
diversity_history = []  # Track population diversity
map_difficulty_history = []  # Track map difficulty per generation

# Load initial map (generation 0)
current_map_difficulty = load_new_map()
print(f"\nGeneration 0: New map loaded (difficulty={current_map_difficulty})\n")

agents = run_generation(POP_SIZE, generation_number=0, map_difficulty=current_map_difficulty)

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
    # Calculate and track population diversity
    diversity_history.append(calculate_diversity(agents))
    # Track map difficulty
    map_difficulty_history.append(current_map_difficulty)

    # Load new map every 3 generations
    if gen % 4 == 0:
        current_map_difficulty = load_new_map()
        print(f"\n  >> New map loaded for generation {gen} (difficulty={current_map_difficulty})\n")

    agents = evolve(agents)
    for agent in agents:
        agent.reset()
    # Run the evolved agents (not random new ones!)
    agents = run_generation_with_agents(agents, generation_number=gen, map_difficulty=current_map_difficulty)

# Save the plot after all generations complete
save_fitness_plot(generation_numbers, avg_distances_history, max_distances_history, survival_count_history, diversity_history, map_difficulty_history)

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
