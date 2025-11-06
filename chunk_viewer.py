"""
Chunk Viewer for BoxJumpEvolve

Visual tool to preview all available obstacle chunks in a grid layout with pagination.
Shows chunk name, difficulty, and visual representation.
Chunks are automatically scaled to fit the screen.

Pagination:
- Page 0: Base chunks (no suffix or ending with _left/_center/_right)
- Page 1: _2 variants
- Page 2: _3 variants
- etc.

Use LEFT/RIGHT arrow keys to navigate between pages.
"""

import pygame
import sys
import re
from maps import get_available_chunk_types

# Initialize Pygame
pygame.init()

# Window settings - Fixed size that fits on most screens
MAX_SCREEN_WIDTH = 1400
MAX_SCREEN_HEIGHT = 900
TITLE_BAR_HEIGHT = 50
GRID_COLS = 4
SEPARATOR_WIDTH = 5
PADDING = 20  # Horizontal padding within each chunk cell

# Get all chunk types
all_chunk_types = get_available_chunk_types()

# Organize chunks by base name and variant number
# Base name extraction: remove _2, _3, etc. and _left/_center/_right suffixes
def get_base_name_and_variant(chunk_name):
    """
    Extract base name and variant number from chunk name.

    Examples:
    - 'single_wall' -> ('single_wall', 0)
    - 'single_wall_2' -> ('single_wall', 2)
    - 'platform_bridge_left' -> ('platform_bridge', 0)
    - 'platform_bridge_left_2' -> ('platform_bridge', 2)
    - 'platform_bridge_center' -> ('platform_bridge', 0)
    - 'staircase_walls_left' -> ('staircase_walls', 0)
    - 'rhythm_jumps_3' -> ('rhythm_jumps', 3)
    """
    # Check if name ends with _<number>
    match = re.search(r'_(\d+)$', chunk_name)
    if match:
        variant_num = int(match.group(1))
        base_name = chunk_name[:match.start()]
        # Remove _left, _center, _right from base name if present
        for suffix in ['_left', '_center', '_right']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        return (base_name, variant_num)
    else:
        # No numeric suffix - check for _left/_center/_right and treat as base
        base_name = chunk_name
        for suffix in ['_left', '_center', '_right']:
            if base_name.endswith(suffix):
                base_name = base_name[:-len(suffix)]
                break
        return (base_name, 0)

# Organize chunks into pages
# Structure: { base_name: { variant_num: chunk_dict } }
chunks_by_base = {}
for chunk in all_chunk_types:
    base_name, variant_num = get_base_name_and_variant(chunk['name'])
    if base_name not in chunks_by_base:
        chunks_by_base[base_name] = {}
    chunks_by_base[base_name][variant_num] = chunk

# Determine max variant number (for number of pages)
max_variant = 0
for variants in chunks_by_base.values():
    max_variant = max(max_variant, max(variants.keys()))

num_pages = max_variant + 1  # Pages 0, 1, 2, ...

# Create a consistent grid ordering (alphabetical by base name)
base_names_ordered = sorted(chunks_by_base.keys())
num_base_chunks = len(base_names_ordered)
grid_rows = (num_base_chunks + GRID_COLS - 1) // GRID_COLS  # Ceiling division

# Calculate cell dimensions to fit on screen
available_width = MAX_SCREEN_WIDTH - (GRID_COLS + 1) * SEPARATOR_WIDTH
available_height = MAX_SCREEN_HEIGHT - TITLE_BAR_HEIGHT - (grid_rows + 1) * SEPARATOR_WIDTH

CELL_WIDTH = available_width // GRID_COLS
CELL_HEIGHT = available_height // grid_rows

# Adjust screen size to actual used space
SCREEN_WIDTH = GRID_COLS * CELL_WIDTH + (GRID_COLS + 1) * SEPARATOR_WIDTH
SCREEN_HEIGHT = grid_rows * CELL_HEIGHT + (grid_rows + 1) * SEPARATOR_WIDTH + TITLE_BAR_HEIGHT

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("BoxJumpEvolve - Chunk Viewer")
clock = pygame.time.Clock()

# Fonts
font_title = pygame.font.SysFont(None, 20, bold=True)
font_info = pygame.font.SysFont(None, 16)

# Colors
COLOR_BG = (240, 240, 240)
COLOR_SEPARATOR = (100, 100, 100)
COLOR_CHUNK_BG = (255, 255, 255)
COLOR_GROUND = (0, 0, 0)
COLOR_GAP = (255, 255, 255)
COLOR_WALL = (100, 100, 100)
COLOR_PLATFORM = (150, 75, 0)
COLOR_TEXT = (0, 0, 0)
COLOR_EMPTY_CELL = (220, 220, 220)

# Ground level for chunk display (within each chunk cell)
GROUND_Y_RATIO = 0.8  # Ground is at 80% down the cell height


def draw_chunk(chunk_type, cell_x, cell_y, cell_width, cell_height):
    """Draw a single chunk in its grid cell, scaled to fit."""
    # Draw chunk background
    chunk_rect = pygame.Rect(cell_x, cell_y, cell_width, cell_height)
    pygame.draw.rect(screen, COLOR_CHUNK_BG, chunk_rect)

    # Generate chunk at start position
    chunk_data = chunk_type['generator'](0)

    # Calculate chunk bounds
    all_x_coords = []
    all_y_coords = []

    for gx, gw in chunk_data['gaps']:
        all_x_coords.extend([gx, gx + gw])
        all_y_coords.append(0)  # Gaps are at ground level

    for wx, ww, wh in chunk_data['walls']:
        all_x_coords.extend([wx, wx + ww])
        all_y_coords.extend([0, -wh])  # Walls extend up from ground

    for px, ph, pw in chunk_data['platforms']:
        all_x_coords.extend([px, px + pw])
        all_y_coords.append(-ph)  # Platforms are above ground

    if not all_x_coords:
        all_x_coords = [0, chunk_type['width']]
        all_y_coords = [0, 0]

    min_x = min(all_x_coords)
    max_x = max(all_x_coords)
    min_y = min(all_y_coords)
    max_y = max(all_y_coords)

    chunk_width = max_x - min_x
    chunk_height = max_y - min_y

    # Calculate scale to fit in cell (with padding)
    drawing_area_width = cell_width - PADDING * 2
    drawing_area_height = cell_height - 60  # Reserve 60px for text at top

    if chunk_width > 0 and chunk_height > 0:
        scale_x = drawing_area_width / chunk_width
        scale_y = drawing_area_height / chunk_height
        scale = min(scale_x, scale_y, 1.0)  # Don't scale up, only down
    else:
        scale = 1.0

    # Calculate drawing position (centered in cell)
    ground_y = cell_y + cell_height * GROUND_Y_RATIO
    center_x = cell_x + cell_width / 2

    # Offset to center the chunk
    offset_x = center_x - (min_x + chunk_width / 2) * scale
    offset_y = ground_y

    # Draw ground line (scaled to chunk's actual width)
    actual_chunk_width = chunk_type['width'] * scale
    ground_start_x = center_x - actual_chunk_width / 2
    ground_end_x = center_x + actual_chunk_width / 2
    pygame.draw.line(screen, (200, 200, 200),
                    (ground_start_x, ground_y),
                    (ground_end_x, ground_y), 2)

    # Draw gaps (white rectangles over ground)
    for gx, gw in chunk_data['gaps']:
        gap_x = offset_x + gx * scale
        gap_w = gw * scale
        pygame.draw.rect(screen, COLOR_GAP,
                        (gap_x, ground_y, gap_w, 30))
        # Draw gap outline to make it visible
        pygame.draw.rect(screen, (200, 0, 0),
                        (gap_x, ground_y, gap_w, 30), 1)

    # Draw walls
    for wx, ww, wh in chunk_data['walls']:
        wall_x = offset_x + wx * scale
        wall_w = ww * scale
        wall_h = wh * scale
        pygame.draw.rect(screen, COLOR_WALL,
                        (wall_x, ground_y - wall_h, wall_w, wall_h))

    # Draw platforms
    for px, ph, pw in chunk_data['platforms']:
        platform_x = offset_x + px * scale
        platform_w = pw * scale
        platform_h = ph * scale
        pygame.draw.rect(screen, COLOR_PLATFORM,
                        (platform_x, ground_y - platform_h, platform_w, 10))

    # Draw chunk info at top
    name_text = font_title.render(chunk_type['name'], True, COLOR_TEXT)
    screen.blit(name_text, (cell_x + 10, cell_y + 5))

    # Draw difficulty
    difficulty = chunk_type['difficulty']
    diff_color = (0, 150, 0) if difficulty <= 3 else (200, 100, 0) if difficulty <= 6 else (200, 0, 0)
    diff_text = font_info.render(f"Diff: {difficulty}", True, diff_color)
    screen.blit(diff_text, (cell_x + 10, cell_y + 25))

    # Draw width
    width_text = font_info.render(f"W: {chunk_type['width']}px", True, (100, 100, 100))
    screen.blit(width_text, (cell_x + 10, cell_y + 40))


def draw_empty_cell(cell_x, cell_y, cell_width, cell_height, base_name):
    """Draw an empty cell when no variant exists for this page."""
    chunk_rect = pygame.Rect(cell_x, cell_y, cell_width, cell_height)
    pygame.draw.rect(screen, COLOR_EMPTY_CELL, chunk_rect)

    # Draw "No variant" text
    text = font_info.render("(no variant)", True, (150, 150, 150))
    text_rect = text.get_rect(center=(cell_x + cell_width // 2, cell_y + cell_height // 2))
    screen.blit(text, text_rect)

    # Show base name faintly
    base_text = font_info.render(base_name, True, (180, 180, 180))
    screen.blit(base_text, (cell_x + 10, cell_y + 10))


def draw_page(page_num):
    """Draw all chunks for the given page."""
    screen.fill(COLOR_BG)

    for grid_idx, base_name in enumerate(base_names_ordered):
        # Calculate grid position
        row = grid_idx // GRID_COLS
        col = grid_idx % GRID_COLS

        # Calculate cell position
        cell_x = col * CELL_WIDTH + (col + 1) * SEPARATOR_WIDTH
        cell_y = row * CELL_HEIGHT + (row + 1) * SEPARATOR_WIDTH + TITLE_BAR_HEIGHT

        # Check if this base chunk has a variant for this page
        variants = chunks_by_base[base_name]
        if page_num in variants:
            chunk_type = variants[page_num]
            draw_chunk(chunk_type, cell_x, cell_y, CELL_WIDTH, CELL_HEIGHT)
        else:
            # Empty cell
            draw_empty_cell(cell_x, cell_y, CELL_WIDTH, CELL_HEIGHT, base_name)

    # Draw title bar at top
    title_bg = pygame.Rect(0, 0, SCREEN_WIDTH, TITLE_BAR_HEIGHT)
    pygame.draw.rect(screen, (60, 60, 60), title_bg)

    # Title with page info
    page_suffix = "" if page_num == 0 else f"_{page_num}"
    title_text = font_title.render(
        f"Obstacle Chunks - Page {page_num}/{max_variant} (showing variants{page_suffix}) - {num_base_chunks} base chunks",
        True, (255, 255, 255)
    )
    screen.blit(title_text, (10, 7))

    # Navigation hint
    hint_text = font_info.render(
        "Use LEFT/RIGHT arrows to navigate pages | ESC to quit",
        True, (200, 200, 200)
    )
    screen.blit(hint_text, (10, 28))


# Main loop
current_page = 0
running = True
draw_page(current_page)
pygame.display.flip()

while running:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_LEFT:
                # Previous page
                current_page = max(0, current_page - 1)
                draw_page(current_page)
                pygame.display.flip()
            elif event.key == pygame.K_RIGHT:
                # Next page
                current_page = min(max_variant, current_page + 1)
                draw_page(current_page)
                pygame.display.flip()

pygame.quit()
sys.exit()
