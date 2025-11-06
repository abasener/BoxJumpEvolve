"""
Map Viewer for BoxJumpEvolve

Visual testing tool to preview map layouts before running full training.
Features:
- Dropdown menu to select map difficulty
- Press 'R' to regenerate RANDOM maps
- Press 'ESC' to quit
"""

import pygame
import sys
from maps import load_map

# Initialize Pygame
pygame.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 1400, 400
GROUND_Y = SCREEN_HEIGHT - 50
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
clock = pygame.time.Clock()
pygame.display.set_caption("BoxJumpEvolve - Map Viewer")

# Fonts
font_title = pygame.font.SysFont(None, 36)
font_dropdown = pygame.font.SysFont(None, 28)
font_help = pygame.font.SysFont(None, 24)

# Colors
COLOR_BG = (255, 255, 255)
COLOR_GROUND = (0, 0, 0)
COLOR_GAP = (255, 255, 255)
COLOR_WALL = (100, 100, 100)
COLOR_PLATFORM = (150, 75, 0)
COLOR_GOAL = (0, 200, 0)
COLOR_DROPDOWN_BG = (230, 230, 230)
COLOR_DROPDOWN_HOVER = (200, 200, 200)
COLOR_DROPDOWN_SELECTED = (180, 180, 255)
COLOR_TEXT = (0, 0, 0)

# Map selection
DIFFICULTIES = ["BASELINE", "MEDIUM", "HARD", "RANDOM"]
current_difficulty_index = 0
current_difficulty = DIFFICULTIES[current_difficulty_index]

# Dropdown state
dropdown_open = False
dropdown_rect = pygame.Rect(20, 20, 200, 40)
dropdown_option_height = 35

# Goal position
GOAL_X = SCREEN_WIDTH - 100

# Load initial map
level_data = load_map(current_difficulty, SCREEN_WIDTH)
gaps = level_data['gaps']
walls = level_data['walls']
platforms = level_data['platforms']


def draw_level():
    """Draw the current map layout."""
    screen.fill(COLOR_BG)

    # Ground
    pygame.draw.rect(screen, COLOR_GROUND, (0, GROUND_Y, SCREEN_WIDTH, 50))

    # Gaps (draw white rectangles over ground)
    for gx, gw in gaps:
        pygame.draw.rect(screen, COLOR_GAP, (gx, GROUND_Y, gw, 50))

    # Walls
    for wx, ww, wh in walls:
        pygame.draw.rect(screen, COLOR_WALL, (wx, GROUND_Y - wh, ww, wh))

    # Platforms
    for px, ph, pw in platforms:
        pygame.draw.rect(screen, COLOR_PLATFORM, (px, GROUND_Y - ph, pw, 10))

    # Goal line
    pygame.draw.line(screen, COLOR_GOAL, (GOAL_X, 0), (GOAL_X, SCREEN_HEIGHT), 3)

    # Add distance markers every 100px for reference
    for x in range(100, SCREEN_WIDTH, 100):
        pygame.draw.line(screen, (200, 200, 200), (x, GROUND_Y - 5), (x, GROUND_Y + 5), 1)
        label = font_help.render(str(x), True, (150, 150, 150))
        screen.blit(label, (x - 10, GROUND_Y + 10))


def draw_dropdown():
    """Draw the difficulty selection dropdown."""
    # Main dropdown button
    if dropdown_open:
        bg_color = COLOR_DROPDOWN_SELECTED
    else:
        mouse_pos = pygame.mouse.get_pos()
        if dropdown_rect.collidepoint(mouse_pos):
            bg_color = COLOR_DROPDOWN_HOVER
        else:
            bg_color = COLOR_DROPDOWN_BG

    pygame.draw.rect(screen, bg_color, dropdown_rect)
    pygame.draw.rect(screen, COLOR_TEXT, dropdown_rect, 2)

    # Current selection text
    text = font_dropdown.render(f"Map: {current_difficulty}", True, COLOR_TEXT)
    text_rect = text.get_rect(center=dropdown_rect.center)
    screen.blit(text, text_rect)

    # Dropdown arrow
    arrow_x = dropdown_rect.right - 20
    arrow_y = dropdown_rect.centery
    if dropdown_open:
        # Up arrow
        pygame.draw.polygon(screen, COLOR_TEXT, [
            (arrow_x, arrow_y + 5),
            (arrow_x - 5, arrow_y - 5),
            (arrow_x + 5, arrow_y - 5)
        ])
    else:
        # Down arrow
        pygame.draw.polygon(screen, COLOR_TEXT, [
            (arrow_x, arrow_y + 5),
            (arrow_x - 5, arrow_y - 5),
            (arrow_x + 5, arrow_y - 5)
        ])

    # Dropdown options (if open)
    if dropdown_open:
        for i, difficulty in enumerate(DIFFICULTIES):
            option_rect = pygame.Rect(
                dropdown_rect.x,
                dropdown_rect.bottom + i * dropdown_option_height,
                dropdown_rect.width,
                dropdown_option_height
            )

            # Hover effect
            mouse_pos = pygame.mouse.get_pos()
            if option_rect.collidepoint(mouse_pos):
                bg_color = COLOR_DROPDOWN_HOVER
            else:
                bg_color = COLOR_DROPDOWN_BG

            pygame.draw.rect(screen, bg_color, option_rect)
            pygame.draw.rect(screen, COLOR_TEXT, option_rect, 1)

            # Option text
            text = font_dropdown.render(difficulty, True, COLOR_TEXT)
            text_rect = text.get_rect(center=option_rect.center)
            screen.blit(text, text_rect)


def draw_help_text():
    """Draw help text at bottom of screen."""
    help_texts = [
        "Press 'R' to regenerate RANDOM map",
        "Press 'ESC' to quit",
        "Click dropdown to change map"
    ]

    y_offset = SCREEN_HEIGHT - 80
    for text_str in help_texts:
        text = font_help.render(text_str, True, (100, 100, 100))
        screen.blit(text, (SCREEN_WIDTH - 350, y_offset))
        y_offset += 25


def draw_map_info():
    """Draw information about the current map."""
    info_y = 80

    # Get difficulty score if available
    difficulty_score = level_data.get('difficulty', 'N/A')

    info_texts = [
        f"Gaps: {len(gaps)}",
        f"Walls: {len(walls)}",
        f"Platforms: {len(platforms)}",
        f"Difficulty: {difficulty_score}"
    ]

    for text_str in info_texts:
        text = font_help.render(text_str, True, COLOR_TEXT)
        screen.blit(text, (20, info_y))
        info_y += 25


def reload_map():
    """Reload the current map (useful for RANDOM maps)."""
    global gaps, walls, platforms, level_data
    level_data = load_map(current_difficulty, SCREEN_WIDTH)
    gaps = level_data['gaps']
    walls = level_data['walls']
    platforms = level_data['platforms']


def handle_dropdown_click(mouse_pos):
    """Handle clicks on the dropdown menu."""
    global current_difficulty_index, current_difficulty, dropdown_open

    # Click on main dropdown button
    if dropdown_rect.collidepoint(mouse_pos):
        dropdown_open = not dropdown_open
        return

    # Click on dropdown options (if open)
    if dropdown_open:
        for i, difficulty in enumerate(DIFFICULTIES):
            option_rect = pygame.Rect(
                dropdown_rect.x,
                dropdown_rect.bottom + i * dropdown_option_height,
                dropdown_rect.width,
                dropdown_option_height
            )
            if option_rect.collidepoint(mouse_pos):
                current_difficulty_index = i
                current_difficulty = DIFFICULTIES[i]
                dropdown_open = False
                reload_map()
                return

    # Click outside dropdown - close it
    dropdown_open = False


# Main loop
running = True
while running:
    clock.tick(30)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_r:
                # Regenerate map (useful for RANDOM)
                reload_map()

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:  # Left click
                handle_dropdown_click(event.pos)

    # Draw everything
    draw_level()
    draw_dropdown()
    draw_help_text()
    draw_map_info()

    pygame.display.flip()

pygame.quit()
sys.exit()
