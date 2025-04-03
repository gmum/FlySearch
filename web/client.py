# I am bad at PyGame, but ChatGPT is kinda better.
# This was vibe-coded away.

import pygame
import requests
import base64
import io
from PIL import Image

# Constants
WIDTH, HEIGHT = 1000, 800
API_URL = "http://localhost:8000"

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent UI")
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()
input_box = pygame.Rect(10, HEIGHT - 60, 200, 40)
send_button = pygame.Rect(220, HEIGHT - 60, 100, 40)
found_button = pygame.Rect(340, HEIGHT - 60, 150, 40)
instruction_text = "Format: x y z (e.g., 1 0 -1)"
user_text = ""


def get_observation():
    response = requests.get(f"{API_URL}/get_observation").json()
    altitude = response["altitude"]
    collision = response["collision"]
    image_data = base64.b64decode(response["image_b64"])  # Decode image
    image = Image.open(io.BytesIO(image_data))
    return altitude, collision, pygame.image.fromstring(image.tobytes(), image.size, image.mode)


def send_action(found, dx, dy, dz):
    requests.post(f"{API_URL}/move", json={"found": found, "coordinate_change": [dx, dy, dz]})


def reset_scenario():
    requests.post(f"{API_URL}/generate_new")


# Initialize scenario on startup
reset_scenario()

# Main loop
running = True
altitude, collision, image = get_observation()
while running:
    screen.fill((0, 0, 0))

    # Draw image
    screen.blit(pygame.transform.scale(image, (WIDTH, HEIGHT - 100)), (0, 0))

    # Display altitude and collision status
    altitude_text = font.render(f"Altitude: {altitude}", True, (255, 255, 255))
    collision_text = font.render(f"Collision: {collision}", True, (255, 0, 0) if collision else (0, 255, 0))
    screen.blit(altitude_text, (10, HEIGHT - 90))
    screen.blit(collision_text, (10, HEIGHT - 130))

    # Draw input box
    pygame.draw.rect(screen, (255, 255, 255), input_box, 2)
    text_surface = font.render(user_text, True, (255, 255, 255))
    screen.blit(text_surface, (input_box.x + 5, input_box.y + 5))

    # Draw instruction text
    instruction_surface = font.render(instruction_text, True, (200, 200, 200))
    screen.blit(instruction_surface, (10, HEIGHT - 160))

    # Draw send button
    pygame.draw.rect(screen, (0, 255, 0), send_button)
    send_text = font.render("SEND", True, (0, 0, 0))
    screen.blit(send_text, (send_button.x + 25, send_button.y + 5))

    # Draw found button
    pygame.draw.rect(screen, (255, 0, 0), found_button)
    found_text = font.render("FOUND", True, (255, 255, 255))
    screen.blit(found_text, (found_button.x + 25, found_button.y + 5))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_RETURN:
                try:
                    dx, dy, dz = map(int, user_text.split())
                    send_action(False, dx, dy, dz)
                    user_text = ""
                except ValueError:
                    user_text = "Invalid input"
            elif event.key == pygame.K_SPACE:
                if not input_box.collidepoint(pygame.mouse.get_pos()):
                    send_action(True, 0, 0, 0)  # Claim found
                    reset_scenario()
                else:
                    user_text += " "
            elif event.key == pygame.K_r:
                reset_scenario()
            elif event.key == pygame.K_BACKSPACE:
                user_text = user_text[:-1]
            else:
                user_text += event.unicode
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if send_button.collidepoint(event.pos):
                try:
                    dx, dy, dz = map(int, user_text.split())
                    send_action(False, dx, dy, dz)
                    user_text = ""
                except ValueError:
                    user_text = "Invalid input"
            elif found_button.collidepoint(event.pos):
                send_action(True, 0, 0, 0)  # Claim found
                reset_scenario()

    pygame.display.flip()
    altitude, collision, image = get_observation()
    clock.tick(10)

pygame.quit()
