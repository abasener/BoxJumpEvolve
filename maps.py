"""
Map definitions and generation for BoxJumpEvolve.

Maps are defined with three obstacle types:
- gaps: (x, width) - deadly gaps in the ground
- walls: (x, width, height) - solid obstacles that block movement
- platforms: (x, height, width) - elevated surfaces to land on
"""

import random

# Standard gap size for physics calculations
STANDARD_GAP_WIDTH = 60

# ============================================================================
# JUMP PHYSICS CONSTANTS (from neuroevo_game.py)
# ============================================================================
# WARNING: These values are hardcoded to match the game physics.
# Changing these may break existing chunk designs that rely on precise jumps!
#
# If you need to modify physics, update BOTH files:
# - neuroevo_game.py (the actual game physics)
# - maps.py (these constants for chunk calculations)
# ============================================================================
BASE_SPEED = 5          # Horizontal pixels per frame at 30 FPS
GRAVITY = 0.5           # Gravity acceleration per frame
JUMP_VELOCITY = -9.0    # Initial vertical velocity when jumping (negative = up)
                        # Calculated as: -(GRAVITY * (1.5 * STANDARD_GAP_WIDTH) / BASE_SPEED)

# Calculate maximum jump distance (horizontal distance agent can clear)
# Using kinematic equations: time_in_air = 2 * abs(JUMP_VELOCITY) / GRAVITY
time_in_air = 2 * abs(JUMP_VELOCITY) / GRAVITY  # 36 frames in air
MAX_JUMP_DISTANCE = BASE_SPEED * time_in_air    # 180 pixels max horizontal distance

# "Barely make it" jump distance (requiring perfect timing)
# This is slightly less than max to account for landing mechanics
PERFECT_JUMP_DISTANCE = int(MAX_JUMP_DISTANCE * 0.95)  # ~171 pixels

# Comfortable jump distance (plenty of room for error)
COMFORTABLE_JUMP_DISTANCE = int(MAX_JUMP_DISTANCE * 0.7)  # ~126 pixels
# ============================================================================


def load_map(difficulty, screen_width=1400, difficulty_mean=3, spread_left=2, spread_right=2):
    """
    Load level elements based on difficulty.

    Args:
        difficulty: "BASELINE", "MEDIUM", "HARD", or "RANDOM"
        screen_width: Width of the screen for map generation
        difficulty_mean: For RANDOM maps, mean difficulty (default 3)
        spread_left: For RANDOM maps, spread below mean (default 2)
        spread_right: For RANDOM maps, spread above mean (default 2)

    Returns:
        dict with 'gaps', 'walls', 'platforms', and 'difficulty' score
    """
    if difficulty == "BASELINE":
        # Original level - basic obstacle course
        # Difficulty: 2 basic gaps + 1 wall + 1 platform = ~5 difficulty
        return {
            'gaps': [(300, STANDARD_GAP_WIDTH), (900, STANDARD_GAP_WIDTH)],
            'walls': [(600, 30, 40)],  # (x, width, height)
            'platforms': [(1100, 40, 60)],  # (x, height, width)
            'difficulty': 5
        }

    elif difficulty == "MEDIUM":
        # Medium - teaches consequence of missing jumps
        # Features: Platform over gap, wall-jump combo, narrow landing
        # Difficulty: 1 basic + 1 wide gap requiring platform + 1 wall = ~12 difficulty
        return {
            'gaps': [
                (250, STANDARD_GAP_WIDTH),      # Gap 1: basic jump
                (650, 120),                      # Gap 2: WIDE - must use platform to cross
                (1050, STANDARD_GAP_WIDTH)       # Gap 3: basic jump after platform
            ],
            'walls': [
                (450, 30, 40)                    # (x, width, height) Wall before wide gap
            ],
            'platforms': [
                (700, 45, 80)                    # Platform in middle of wide gap - must land here
            ],
            'difficulty': 12
        }

    elif difficulty == "HARD":
        # Hard - requires rhythm, precision, and TIMING
        # Features: Triple jump, drop-and-wait trap, timing puzzle, false platform
        # Difficulty: Triple rhythm (6) + trap mechanics (8) + timing (7) + false platform (7) = ~28 difficulty
        return {
            'gaps': [
                (200, 50),                       # Gap 1: narrow
                (350, 50),                       # Gap 2: narrow - quick double jump!
                (500, 50),                       # Gap 3: narrow - triple jump rhythm!
                (790, 70),                       # Gap 4: TRAP GAP - after tall wall, if you jump you overshoot platform into this!
                (920, 60),                       # Gap 5: After safe platform - need to time jump correctly
                (1050, 50)                       # Gap 6: Final gap to goal
            ],
            'walls': [
                (700, 20, 40),                   # (x, width, height) Wall 1 (short)
                (730, 20, 80)                    # (x, width, height) Wall 2 (TALL - staircase) - must fall carefully after!
            ],
            'platforms': [
                (870, 30, 40),                   # DROP-AND-WAIT platform - if you jump from wall, you overshoot into gap 4!
                (1000, 35, 40),                  # TIMING platform - must land on first platform, wait, then jump to this
                (600, 70, 30)                    # FALSE PLATFORM - high trap that leads nowhere, punishes random jumping
            ],
            'difficulty': 28
        }

    elif difficulty == "RANDOM":
        # Random map generation using obstacle chunks with difficulty distribution
        return generate_random_map(screen_width, difficulty_mean, spread_left, spread_right)

    else:
        raise ValueError(f"Unknown difficulty: {difficulty}")


def sample_difficulty(mean, spread_left, spread_right):
    """
    Sample a difficulty value from an asymmetric normal distribution.

    Args:
        mean: Center of the distribution
        spread_left: Standard deviation for values below the mean
        spread_right: Standard deviation for values above the mean

    Returns:
        Integer difficulty value (clamped to 1-10 range)

    Example:
        mean=7, spread_left=6, spread_right=2
        - Most samples will be around 7
        - ~68% will be in [7-6, 7+2] = [1, 9]
        - Can occasionally go as low as 1 or as high as 10
    """
    # Sample from standard normal distribution
    z = random.gauss(0, 1)

    # Use different spread depending on sign
    if z < 0:
        # Below mean - use spread_left
        value = mean + z * spread_left
    else:
        # Above mean - use spread_right
        value = mean + z * spread_right

    # Clamp to valid difficulty range [1, 10] and round to integer
    return max(1, min(10, round(value)))


def generate_random_map(screen_width, difficulty_mean=3, spread_left=2, spread_right=2):
    """
    Generate a random map using obstacle chunks with difficulty distribution control.

    Algorithm:
    1. Define safety buffer at start for agents to read inputs
    2. Calculate available space from buffer to goal
    3. Sample chunk difficulties from asymmetric normal distribution
    4. For each sampled difficulty, randomly select a chunk with that difficulty
    5. Repeat until total width exceeds available space
    6. Remove the last chunk that exceeded space
    7. Distribute remaining whitespace randomly between chunks

    Args:
        screen_width: Width of the screen
        difficulty_mean: Mean difficulty for the normal distribution (default 3 = easy-medium)
        spread_left: Standard deviation below the mean (default 2)
        spread_right: Standard deviation above the mean (default 2)
            Example: (mean=7, spread_left=6, spread_right=2) = mostly hard with few easy breaks
            Example: (mean=2, spread_left=1, spread_right=3) = mostly easy with occasional challenges

    Returns:
        dict with 'gaps', 'walls', 'platforms', and 'difficulty' score
    """
    START_X = 50
    GOAL_X = screen_width - 100
    SAFETY_BUFFER = 150  # Pixels of safe space at start

    # Available space for obstacles
    start_position = START_X + SAFETY_BUFFER
    available_space = GOAL_X - start_position

    # Get available chunk types and organize by difficulty
    chunk_types = get_available_chunk_types()

    if not chunk_types:
        # No chunks defined yet - return empty map
        return {
            'gaps': [],
            'walls': [],
            'platforms': [],
            'difficulty': 0
        }

    # Organize chunks by difficulty level for quick lookup
    chunks_by_difficulty = {}
    for chunk in chunk_types:
        diff = chunk['difficulty']
        if diff not in chunks_by_difficulty:
            chunks_by_difficulty[diff] = []
        chunks_by_difficulty[diff].append(chunk)

    # Randomly select chunks until we exceed available space
    selected_chunks = []
    total_width = 0
    total_difficulty = 0
    max_attempts = 1000  # Prevent infinite loops

    for _ in range(max_attempts):
        if total_width >= available_space:
            break

        # Sample a difficulty from the asymmetric normal distribution
        target_difficulty = sample_difficulty(difficulty_mean, spread_left, spread_right)

        # Find chunks with this exact difficulty, or closest available
        if target_difficulty in chunks_by_difficulty:
            available_chunks = chunks_by_difficulty[target_difficulty]
        else:
            # Find closest difficulty level that has chunks
            closest_diff = min(chunks_by_difficulty.keys(), key=lambda d: abs(d - target_difficulty))
            available_chunks = chunks_by_difficulty[closest_diff]

        # Randomly pick one chunk with this difficulty
        chunk_type = random.choice(available_chunks)
        chunk_width = chunk_type['width']
        chunk_difficulty = chunk_type['difficulty']

        # Add chunk
        selected_chunks.append(chunk_type)
        total_width += chunk_width
        total_difficulty += chunk_difficulty

    # Remove the last chunk that caused us to exceed space
    if selected_chunks:
        last_chunk = selected_chunks.pop()
        total_width -= last_chunk['width']
        total_difficulty -= last_chunk['difficulty']

    # Calculate whitespace to distribute
    whitespace = available_space - total_width

    # Distribute whitespace randomly between chunks, start, and end
    num_gaps = len(selected_chunks) + 1  # gaps before each chunk + one at end
    spacing = distribute_whitespace(whitespace, num_gaps)

    # Generate actual obstacles by placing chunks with spacing
    all_gaps = []
    all_walls = []
    all_platforms = []

    current_x = start_position + spacing[0]  # Start with first spacing

    for i, chunk in enumerate(selected_chunks):
        # Generate chunk at current_x position
        chunk_data = chunk['generator'](current_x)

        # Add chunk's obstacles to our lists
        all_gaps.extend(chunk_data['gaps'])
        all_walls.extend(chunk_data['walls'])
        all_platforms.extend(chunk_data['platforms'])

        # Move to next chunk position (chunk width + next spacing)
        current_x += chunk['width'] + spacing[i + 1]

    return {
        'gaps': all_gaps,
        'walls': all_walls,
        'platforms': all_platforms,
        'difficulty': total_difficulty
    }


def get_available_chunk_types():
    """
    Return list of available chunk types.

    Each chunk type is a dict with:
    - 'name': Human-readable name
    - 'width': Approximate width in pixels
    - 'difficulty': Difficulty score (1-10, higher = harder)
    - 'generator': Function that takes start_x and returns {'gaps': [], 'walls': [], 'platforms': []}

    Difficulty Guidelines:
    - 0: Nothing
    - 1-3: Easy (basic jumps, small obstacles)
    - 4-6: Medium (requires timing, multiple jumps)
    - 7-9: Hard (tricky patterns, precision required)
    - 10: Very Hard for a human (tight timings, counter-intuitive)
    """
    return [
        # Easy chunks (1-3) - No gaps, no death risk
        {'name': 'single_platform', 'width': 80, 'difficulty': 1, 'generator': chunk_single_platform},
        {'name': 'two_platforms', 'width': 150, 'difficulty': 1, 'generator': chunk_two_platforms},
        {'name': 'platform_stairs', 'width': 180, 'difficulty': 1, 'generator': chunk_platform_stairs},
        {'name': 'overlapping_platforms', 'width': 100, 'difficulty': 1, 'generator': chunk_overlapping_platforms},
        {'name': 'multi_level_platforms', 'width': 200, 'difficulty': 1, 'generator': chunk_multi_level_platforms},

        {'name': 'single_wall', 'width': 50, 'difficulty': 2, 'generator': chunk_single_wall},
        {'name': 'single_wall_2', 'width': 50, 'difficulty': 2, 'generator': chunk_single_wall_2},
        {'name': 'wall_and_platform', 'width': 120, 'difficulty': 2, 'generator': chunk_wall_and_platform},
        {'name': 'wall_and_platform_2', 'width': 120, 'difficulty': 3, 'generator': chunk_wall_and_platform_2},

        # Staircase walls split by direction
        {'name': 'staircase_walls_left', 'width': 120, 'difficulty': 3, 'generator': chunk_staircase_walls_left},
        {'name': 'staircase_walls', 'width': 120, 'difficulty': 4, 'generator': chunk_staircase_walls},
        {'name': 'staircase_walls_2', 'width': 120, 'difficulty': 3, 'generator': chunk_staircase_walls_2},

        # Medium chunks (4-6) - Has gaps (death risk) or requires strategy
        {'name': 'single_gap', 'width': 70, 'difficulty': 4, 'generator': chunk_single_gap},
        {'name': 'single_gap_2', 'width': 85, 'difficulty': 5, 'generator': chunk_single_gap_2},

        {'name': 'gentle_gaps', 'width': 180, 'difficulty': 4, 'generator': chunk_gentle_gaps},
        {'name': 'gentle_gaps_2', 'width': 160, 'difficulty': 5, 'generator': chunk_gentle_gaps_2},
        {'name': 'gentle_gaps_3', 'width': 180, 'difficulty': 5, 'generator': chunk_gentle_gaps_3},

        {'name': 'platform_gap_combo', 'width': 150, 'difficulty': 4, 'generator': chunk_platform_gap_combo},
        {'name': 'platform_gap_combo_2', 'width': 170, 'difficulty': 5, 'generator': chunk_platform_gap_combo_2},

        {'name': 'wall_dip', 'width': 120, 'difficulty': 5, 'generator': chunk_wall_dip},
        {'name': 'wall_dip_2', 'width': 100, 'difficulty': 6, 'generator': chunk_wall_dip_2},

        {'name': 'rhythm_jumps', 'width': 250, 'difficulty': 5, 'generator': chunk_rhythm_jumps},
        {'name': 'rhythm_jumps_2', 'width': 220, 'difficulty': 7, 'generator': chunk_rhythm_jumps_2},

        # Platform bridge split by position (left/center/right)
        {'name': 'platform_bridge_left', 'width': 200, 'difficulty': 5, 'generator': chunk_platform_bridge_left},
        {'name': 'platform_bridge_left_2', 'width': 200, 'difficulty': 6, 'generator': chunk_platform_bridge_left_2},
        {'name': 'platform_bridge_center', 'width': 200, 'difficulty': 6, 'generator': chunk_platform_bridge_center},
        {'name': 'platform_bridge_center_2', 'width': 200, 'difficulty': 8, 'generator': chunk_platform_bridge_center_2},
        {'name': 'platform_bridge_right', 'width': 200, 'difficulty': 7, 'generator': chunk_platform_bridge_right},

        {'name': 'wall_dip_fail', 'width': 120, 'difficulty': 7, 'generator': chunk_wall_dip_fail},
        {'name': 'wall_dip_fail_2', 'width': 120, 'difficulty': 6, 'generator': chunk_wall_dip_fail_2},

        # Hard chunks (7-9) - Traps, timing, counter-intuitive
        {'name': 'gap_trap', 'width': 180, 'difficulty': 8, 'generator': chunk_gap_trap},
        {'name': 'gap_trap_2', 'width': 180, 'difficulty': 7, 'generator': chunk_gap_trap_2},

        {'name': 'rhythm_jumps_3', 'width': 530, 'difficulty': 9, 'generator': chunk_rhythm_jumps_3},
    ]


def distribute_whitespace(total_whitespace, num_gaps):
    """
    Randomly distribute whitespace between gaps.

    Args:
        total_whitespace: Total pixels to distribute
        num_gaps: Number of gaps to distribute between

    Returns:
        List of spacing values (one per gap)
    """
    if num_gaps == 0:
        return []

    if num_gaps == 1:
        return [total_whitespace]

    # Use random partitioning: generate random split points
    # This creates varied spacing while using all whitespace
    split_points = sorted([random.uniform(0, total_whitespace) for _ in range(num_gaps - 1)])
    split_points = [0] + split_points + [total_whitespace]

    # Calculate spacing from split points
    spacing = [split_points[i+1] - split_points[i] for i in range(num_gaps)]

    return spacing


# --- Obstacle Chunk Generators (TODO) ---
# These will be implemented incrementally

def chunk_rhythm_jumps(start_x):
    """
    Generate 3 gaps with consistent spacing for rhythm practice.
    Teaches: Jump timing, rhythm
    Difficulty: 3 (Easy-medium - just requires timing)
    Width: ~250px
    """
    gaps = [
        (start_x, 50),           # Gap 1
        (start_x + 100, 50),     # Gap 2
        (start_x + 200, 50),     # Gap 3
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': []}


def chunk_staircase_walls(start_x):
    """
    Generate 3 progressively taller walls (staircase).
    Teaches: Jumping onto elevated surfaces, climbing
    Difficulty: 4 (Medium - requires multiple jumps up)
    Width: ~120px
    """
    walls = [
        (start_x, 20, 30),           # Wall 1: short (30px)
        (start_x + 40, 20, 50),      # Wall 2: medium (50px)
        (start_x + 80, 20, 70),      # Wall 3: tall (70px)
    ]
    return {'gaps': [], 'walls': walls, 'platforms': []}


def chunk_wall_dip(start_x):
    """
    Two walls with a dip between - can jump wall-to-wall or fall and jump out.
    Teaches: Wall jumping, recovery from dips
    Difficulty: 5 (Medium - multiple solution paths)
    Width: ~120px
    """
    walls = [
        (start_x, 20, 60),           # Wall 1: tall
        (start_x + 80, 20, 60),      # Wall 2: tall (60px gap between)
    ]
    return {'gaps': [], 'walls': walls, 'platforms': []}


def chunk_wall_dip_fail(start_x):
    """
    Wall dip trap where falling in means getting stuck (timeout).
    First wall (50px), dip (40px), second wall (80px - too high to jump from dip).
    Teaches: Don't fall into every dip, think ahead
    Difficulty: 7 (Hard - agents get trapped and timeout)
    Width: ~120px
    """
    walls = [
        (start_x, 20, 50),           # Wall 1: medium height
        (start_x + 60, 20, 80),      # Wall 2: very tall (can't jump out from dip)
    ]
    # No gap - agents get stuck between walls and timeout
    return {'gaps': [], 'walls': walls, 'platforms': []}


def chunk_gap_trap(start_x):
    """
    Tall wall followed immediately by trap gap (like HARD map design).
    If you jump from wall, you overshoot platform into gap. Must fall carefully.
    Teaches: Patience, don't always jump
    Difficulty: 8 (Hard - counter-intuitive, punishes jumping)
    Width: ~180px
    """
    walls = [
        (start_x, 20, 80),           # Tall wall
    ]
    gaps = [
        (start_x + 40, 70),          # Trap gap - overshoot if you jump from wall
    ]
    platforms = [
        (start_x + 120, 40, 30),     # Safe platform - only reachable by falling, not jumping
    ]
    return {'gaps': gaps, 'walls': walls, 'platforms': platforms}


def chunk_platform_bridge(start_x):
    """
    Wide gap with platform in the middle - must land on platform to cross.
    Teaches: Precision landing, multi-stage jumps
    Difficulty: 4 (Medium - requires precision)
    Width: ~200px
    """
    gaps = [
        (start_x, 120),              # Wide gap
    ]
    platforms = [
        (start_x + 50, 40, 60),      # Platform in middle of gap
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': platforms}


def chunk_single_platform(start_x):
    """
    Simple elevated platform - very easy, just for variety.
    Teaches: Basic platform landing
    Difficulty: 1 (Easy - trivial obstacle)
    Width: ~80px
    """
    platforms = [
        (start_x, 35, 60),           # Low platform
    ]
    return {'gaps': [], 'walls': [], 'platforms': platforms}


def chunk_single_gap(start_x):
    """
    Single basic gap - easiest obstacle.
    Teaches: Basic jumping
    Difficulty: 1 (Easy - fundamental skill)
    Width: ~70px
    """
    gaps = [
        (start_x, 60),               # Standard gap
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': []}


def chunk_single_wall(start_x):
    """
    Single low wall - can be jumped over easily.
    Teaches: Basic wall jumping
    Difficulty: 1 (Easy - simple wall jump)
    Width: ~50px
    """
    walls = [
        (start_x, 20, 35),           # Low wall (35px - easy to jump over)
    ]
    return {'gaps': [], 'walls': walls, 'platforms': []}


def chunk_two_platforms(start_x):
    """
    Two low platforms in sequence - no gaps, no walls.
    Teaches: Platform landing without danger
    Difficulty: 1 (Easy - safe practice)
    Width: ~150px
    """
    platforms = [
        (start_x, 30, 50),           # Platform 1: low
        (start_x + 80, 35, 50),      # Platform 2: slightly higher
    ]
    return {'gaps': [], 'walls': [], 'platforms': platforms}


def chunk_wall_and_platform(start_x):
    """
    Low wall followed by a platform - safe navigation.
    Teaches: Wall + platform combo without traps
    Difficulty: 2 (Easy - basic combo)
    Width: ~120px
    """
    walls = [
        (start_x, 20, 40),           # Low wall
    ]
    platforms = [
        (start_x + 60, 40, 50),      # Platform after wall
    ]
    return {'gaps': [], 'walls': walls, 'platforms': platforms}


def chunk_gentle_gaps(start_x):
    """
    Two small gaps with safe landing space between.
    Teaches: Multiple jumps with recovery time
    Difficulty: 2 (Easy - basic rhythm, forgiving)
    Width: ~180px
    """
    gaps = [
        (start_x, 50),               # Gap 1: small
        (start_x + 120, 50),         # Gap 2: small (lots of space between)
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': []}


def chunk_platform_stairs(start_x):
    """
    Three platforms at increasing heights - can stay low or climb up.
    Teaches: Platform navigation, optional vertical movement
    Difficulty: 1 (Easy - safe climbing practice)
    Width: ~180px
    """
    platforms = [
        (start_x, 25, 50),           # Platform 1: low (25px)
        (start_x + 60, 40, 50),      # Platform 2: medium (40px)
        (start_x + 120, 55, 50),     # Platform 3: high (55px)
    ]
    return {'gaps': [], 'walls': [], 'platforms': platforms}


def chunk_overlapping_platforms(start_x):
    """
    Two overlapping platforms at different heights - stay low or jump up.
    Teaches: Platform choice, vertical navigation
    Difficulty: 1 (Easy - no danger, just choice)
    Width: ~100px
    """
    platforms = [
        (start_x, 30, 60),           # Platform 1: low, wide
        (start_x + 30, 50, 50),      # Platform 2: higher, overlaps first
    ]
    return {'gaps': [], 'walls': [], 'platforms': platforms}


def chunk_platform_gap_combo(start_x):
    """
    Gap with overlapping platforms on each side - can go high or low.
    Teaches: Platform navigation with gap crossing
    Difficulty: 2 (Easy-medium - gap + platform choice)
    Width: ~150px
    """
    gaps = [
        (start_x + 50, 60),          # Gap in the middle
    ]
    platforms = [
        (start_x, 35, 40),           # Platform before gap (low)
        (start_x + 10, 55, 40),      # Platform before gap (high, overlaps)
        (start_x + 120, 35, 40),     # Platform after gap (low)
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': platforms}


def chunk_multi_level_platforms(start_x):
    """
    Four platforms at varying heights creating multiple paths.
    Teaches: Complex platform navigation, path choice
    Difficulty: 2 (Easy-medium - safe but requires navigation)
    Width: ~200px
    """
    platforms = [
        (start_x, 30, 45),           # Platform 1: low left
        (start_x + 50, 50, 45),      # Platform 2: high middle-left
        (start_x + 100, 35, 45),     # Platform 3: medium middle-right
        (start_x + 150, 45, 45),     # Platform 4: medium-high right
    ]
    return {'gaps': [], 'walls': [], 'platforms': platforms}


# ============================================================================
# CHUNK VARIANTS (difficulty 2+)
# ============================================================================
# Naming convention: original_name_2, original_name_3, etc.
# Split asymmetric chunks use: name_left, name_right, name_center + variants
# ============================================================================

# --- single_wall variants ---
def chunk_single_wall_2(start_x):
    """
    Single shorter wall - easier to jump over.
    Teaches: Basic wall jumping with less height
    Difficulty: 2 (Easy - simpler wall jump)
    Width: ~50px
    """
    walls = [
        (start_x, 20, 25),           # Shorter wall (25px vs original 35px)
    ]
    return {'gaps': [], 'walls': walls, 'platforms': []}


# --- wall_and_platform variants ---
def chunk_wall_and_platform_2(start_x):
    """
    Taller wall with platform closer - tests wall climbing.
    Teaches: Wall + platform combo with more vertical challenge
    Difficulty: 3 (Easy-medium - taller wall)
    Width: ~120px
    """
    walls = [
        (start_x, 20, 55),           # Taller wall
    ]
    platforms = [
        (start_x + 50, 45, 50),      # Platform closer and higher
    ]
    return {'gaps': [], 'walls': walls, 'platforms': platforms}


# --- single_gap variants ---
def chunk_single_gap_2(start_x):
    """
    Wider single gap - requires better timing.
    Teaches: Jumping across wider gaps
    Difficulty: 5 (Medium - wider gap, more risk)
    Width: ~85px
    """
    gaps = [
        (start_x, 75),               # Wider gap (75px vs 60px)
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': []}


# --- gentle_gaps variants ---
def chunk_gentle_gaps_2(start_x):
    """
    Two gaps with less space between - quicker rhythm required.
    Teaches: Faster-paced multiple jumps
    Difficulty: 5 (Medium - less recovery time)
    Width: ~160px
    """
    gaps = [
        (start_x, 50),               # Gap 1: small
        (start_x + 90, 50),          # Gap 2: closer (90px vs 120px spacing)
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': []}

def chunk_gentle_gaps_3(start_x):
    """
    Two gaps, first landing zone shorter - asymmetric timing.
    Teaches: Adapting rhythm to varied landing zones
    Difficulty: 5 (Medium - asymmetric spacing)
    Width: ~180px
    """
    gaps = [
        (start_x, 50),               # Gap 1: small
        (start_x + 80, 50),          # Gap 2: first landing is shorter
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': []}


# --- staircase_walls variants (split into left/right for directionality) ---
def chunk_staircase_walls_left(start_x):
    """
    3 walls descending (tall to short) - going down stairs.
    Teaches: Descending elevated surfaces
    Difficulty: 3 (Easy-medium - easier than climbing)
    Width: ~120px
    """
    walls = [
        (start_x, 20, 70),           # Wall 1: tall (70px)
        (start_x + 40, 20, 50),      # Wall 2: medium (50px)
        (start_x + 80, 20, 30),      # Wall 3: short (30px)
    ]
    return {'gaps': [], 'walls': walls, 'platforms': []}

def chunk_staircase_walls_2(start_x):
    """
    3 progressively taller walls with smaller height gaps - gentler stairs.
    Teaches: Climbing with less dramatic elevation change
    Difficulty: 3 (Easy-medium - gentler climb)
    Width: ~120px
    """
    walls = [
        (start_x, 20, 35),           # Wall 1: short (35px)
        (start_x + 40, 20, 45),      # Wall 2: medium (45px)
        (start_x + 80, 20, 55),      # Wall 3: tall (55px) - smaller steps
    ]
    return {'gaps': [], 'walls': walls, 'platforms': []}


# --- platform_gap_combo variants ---
def chunk_platform_gap_combo_2(start_x):
    """
    Wider gap with platforms - more challenging crossing.
    Teaches: Platform navigation with wider gap
    Difficulty: 5 (Medium - wider gap)
    Width: ~170px
    """
    gaps = [
        (start_x + 50, 80),          # Wider gap
    ]
    platforms = [
        (start_x, 35, 40),           # Platform before gap (low)
        (start_x + 10, 55, 40),      # Platform before gap (high, overlaps)
        (start_x + 140, 35, 40),     # Platform after gap (low) - further away
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': platforms}


# --- wall_dip variants ---
def chunk_wall_dip_2(start_x):
    """
    Two walls with narrower dip - tighter space to navigate.
    Teaches: Wall jumping in tighter spaces
    Difficulty: 6 (Medium-hard - less room for error)
    Width: ~100px
    """
    walls = [
        (start_x, 20, 60),           # Wall 1: tall
        (start_x + 60, 20, 60),      # Wall 2: tall (narrower 40px gap vs 60px)
    ]
    return {'gaps': [], 'walls': walls, 'platforms': []}


# --- rhythm_jumps variants (difficulty 5, 7, 9 versions) ---
def chunk_rhythm_jumps_2(start_x):
    """
    3 gaps with tighter spacing - less ground between jumps.
    Teaches: Jump timing with minimal recovery time
    Difficulty: 7 (Hard - tight rhythm, small landing zones)
    Width: ~220px
    """
    # Ground segments: ~30px each (small landing zones)
    gaps = [
        (start_x, 50),               # Gap 1
        (start_x + 80, 50),          # Gap 2 (30px ground between)
        (start_x + 160, 50),         # Gap 3 (30px ground between)
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': []}

def chunk_rhythm_jumps_3(start_x):
    """
    3 gaps with pixel-perfect spacing - must jump immediately on landing.
    Teaches: Frame-perfect jump rhythm (very hard)
    Difficulty: 9 (Very hard - pixel-perfect timing required)
    Width: ~200px
    """
    # Ground segments: ~15px each (barely enough to land before next jump)
    # Gaps sized just under max jump distance
    gap_width = PERFECT_JUMP_DISTANCE - 10  # Slightly less than perfect jump
    landing_space = 15  # Minimal landing room

    gaps = [
        (start_x, gap_width),                                    # Gap 1
        (start_x + gap_width + landing_space, gap_width),       # Gap 2
        (start_x + 2 * (gap_width + landing_space), gap_width), # Gap 3
    ]
    width = 3 * gap_width + 2 * landing_space
    return {'gaps': gaps, 'walls': [], 'platforms': []}


# --- platform_bridge split into left/center/right + variants ---
def chunk_platform_bridge_left(start_x):
    """
    Wide gap with platform on LEFT side - must jump early.
    Teaches: Early jump timing, landing on near-side platform
    Difficulty: 5 (Medium - easier, platform is closer)
    Width: ~200px
    """
    gaps = [
        (start_x, 120),              # Wide gap
    ]
    platforms = [
        (start_x + 20, 40, 60),      # Platform on left side of gap
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': platforms}

def chunk_platform_bridge_center(start_x):
    """
    Wide gap with platform in CENTER - original design.
    Teaches: Precision landing, multi-stage jumps
    Difficulty: 6 (Medium-hard - requires precision)
    Width: ~200px
    """
    gaps = [
        (start_x, 120),              # Wide gap
    ]
    platforms = [
        (start_x + 50, 40, 60),      # Platform in center
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': platforms}

def chunk_platform_bridge_right(start_x):
    """
    Wide gap with platform on RIGHT side - must jump late/far.
    Teaches: Late jump timing, landing on far-side platform
    Difficulty: 7 (Hard - platform is far, requires precise jump arc)
    Width: ~200px
    """
    gaps = [
        (start_x, 120),              # Wide gap
    ]
    platforms = [
        (start_x + 80, 40, 60),      # Platform on right side of gap
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': platforms}

def chunk_platform_bridge_center_2(start_x):
    """
    Wide gap with NARROW platform in center - precise landing required.
    Teaches: Pixel-perfect platform landing
    Difficulty: 8 (Hard - narrow platform)
    Width: ~200px
    """
    gaps = [
        (start_x, 120),              # Wide gap
    ]
    platforms = [
        (start_x + 50, 40, 35),      # Narrower platform (35px vs 60px)
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': platforms}

def chunk_platform_bridge_left_2(start_x):
    """
    Wide gap with narrow platform on left - early precise jump.
    Teaches: Early precise landing
    Difficulty: 6 (Medium-hard - narrow platform, but closer)
    Width: ~200px
    """
    gaps = [
        (start_x, 120),              # Wide gap
    ]
    platforms = [
        (start_x + 20, 40, 40),      # Narrower platform on left
    ]
    return {'gaps': gaps, 'walls': [], 'platforms': platforms}


# --- wall_dip_fail variants ---
def chunk_wall_dip_fail_2(start_x):
    """
    Wall dip trap with slightly lower second wall - still traps but less obvious.
    Teaches: Don't fall into dips, recognition of trap patterns
    Difficulty: 6 (Medium-hard - less obvious trap)
    Width: ~120px
    """
    walls = [
        (start_x, 20, 50),           # Wall 1: medium height
        (start_x + 60, 20, 70),      # Wall 2: tall but jumpable from dip (70px vs 80px)
    ]
    return {'gaps': [], 'walls': walls, 'platforms': []}


# --- gap_trap variants ---
def chunk_gap_trap_2(start_x):
    """
    Medium wall with closer trap gap - less obvious trap.
    Teaches: Patience even with medium walls
    Difficulty: 7 (Hard - subtler trap)
    Width: ~180px
    """
    walls = [
        (start_x, 20, 60),           # Medium wall (less tall)
    ]
    gaps = [
        (start_x + 35, 70),          # Trap gap - closer to wall
    ]
    platforms = [
        (start_x + 120, 40, 30),     # Safe platform
    ]
    return {'gaps': gaps, 'walls': walls, 'platforms': platforms}


# ============================================================================
# UPDATE get_available_chunk_types() TO INCLUDE ALL VARIANTS
# ============================================================================
