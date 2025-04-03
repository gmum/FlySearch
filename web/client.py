import pygame
import requests
import base64
import io
import time
from PIL import Image

# Constants
WIDTH, HEIGHT = 800, 600
API_URL = "http://localhost:8000"

# Pygame setup
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent UI")
font = pygame.font.Font(None, 36)
clock = pygame.time.Clock()
input_boxes = [
    pygame.Rect(10, HEIGHT - 60, 50, 40),
    pygame.Rect(70, HEIGHT - 60, 50, 40),
    pygame.Rect(130, HEIGHT - 60, 50, 40)
]
current_input = 0
send_button = pygame.Rect(220, HEIGHT - 60, 100, 40)
found_button = pygame.Rect(340, HEIGHT - 60, 150, 40)
instruction_text = "Enter x, y, z separately"
user_inputs = ["", "", ""]
observation = None


def get_observation():
    global observation
    response = requests.get(f"{API_URL}/get_observation").json()
    altitude = response["altitude"]
    collision = response["collision"]
    image_data = base64.b64decode(response["image_b64"])  # Decode image
    image = Image.open(io.BytesIO(image_data))
    observation = (altitude, collision, pygame.image.fromstring(image.tobytes(), image.size, image.mode))


def send_action(found, dx, dy, dz):
    requests.post(f"{API_URL}/move", json={"found": found, "coordinate_change": [dx, dy, dz]})
    if found:
        reset_scenario()
    else:
        time.sleep(1)  # Wait before requesting a new observation
        get_observation()


def reset_scenario():
    requests.post(f"{API_URL}/generate_new")
    get_observation()


# Initialize scenario on startup
reset_scenario()

# Main loop
running = True
while running:
    screen.fill((0, 0, 0))
    altitude, collision, image = observation

    # Draw image
    screen.blit(pygame.transform.scale(image, (WIDTH, HEIGHT - 100)), (0, 0))

    # Display altitude and collision status
    altitude_text = font.render(f"Altitude: {altitude}", True, (255, 255, 255))
    collision_text = font.render(f"Collision: {collision}", True, (255, 0, 0) if collision else (0, 255, 0))
    screen.blit(altitude_text, (10, HEIGHT - 140))
    screen.blit(collision_text, (10, HEIGHT - 100))

    # Draw input boxes
    for i, box in enumerate(input_boxes):
        pygame.draw.rect(screen, (255, 255, 255), box, 2)
        text_surface = font.render(user_inputs[i], True, (255, 255, 255))
        screen.blit(text_surface, (box.x + 5, box.y + 5))

        # Cursor indication
        if i == current_input:
            pygame.draw.line(screen, (255, 255, 255), (box.x + 5 + text_surface.get_width(), box.y + 5),
                             (box.x + 5 + text_surface.get_width(), box.y + 35))

    # Draw instruction text
    instruction_surface = font.render(instruction_text, True, (200, 200, 200))
    screen.blit(instruction_surface, (10, HEIGHT - 180))

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
                    dx, dy, dz = map(int, user_inputs)
                    send_action(False, dx, dy, dz)
                    user_inputs = ["", "", ""]
                except ValueError:
                    user_inputs = ["", "", ""]
            elif event.key == pygame.K_BACKSPACE:
                user_inputs[current_input] = user_inputs[current_input][:-1]
            elif event.key == pygame.K_TAB:
                current_input = (current_input + 1) % 3
            elif event.unicode.isdigit() or (event.unicode == "-" and len(user_inputs[current_input]) == 0):
                user_inputs[current_input] += event.unicode
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if send_button.collidepoint(event.pos):
                try:
                    dx, dy, dz = map(int, user_inputs)
                    send_action(False, dx, dy, dz)
                    user_inputs = ["", "", ""]
                except ValueError:
                    user_inputs = ["", "", ""]
            elif found_button.collidepoint(event.pos):
                send_action(True, 0, 0, 0)  # Claim found
            for i, box in enumerate(input_boxes):
                if box.collidepoint(event.pos):
                    current_input = i

    pygame.display.flip()
    clock.tick(10)

pygame.quit()
