import cv2
import mediapipe as mp
import numpy as np
import pygame
import random
import math
import time
import random

# MediaPipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Screen info
# infoObject = pygame.display.Info()

# PyGame setup
pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand Pong")

WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
BLACK = (0, 0, 0)

# List of balls
balls = []

# Ball properties
ball_radius = 20
ball_x, ball_y = WIDTH // 2, HEIGHT // 4
ball_vx, ball_vy = 0, 5
gravity = 0.75 + abs(ball_vy) * 0.02
bounce_speed = -12.5  # Upward speed after a bounce
max_bounce_speed = -25  # Maximum bounce speed

balls.append({
    "x": ball_x,
    "y": ball_y,
    "vx": ball_vx,
    "vy": ball_vy,
})

# Paddle properties
paddle_width, paddle_height = 180, 145
paddle_x, paddle_y = WIDTH // 2 - paddle_width // 2, HEIGHT - 50
paddle_smoothing = 0.9  # Smoothing factor for paddle movement
smoothed_paddle_x, smoothed_paddle_y = paddle_x, paddle_y
previous_paddle_y = paddle_y
paddle_speed_threshold = 3

# Padle image
paddle_image = pygame.image.load("assets/paddle.png").convert_alpha()
paddle_image = pygame.transform.scale(paddle_image, (paddle_width, paddle_height)) 

# Ball image
ball_image = pygame.image.load("assets/ball.png").convert_alpha()
ball_image = pygame.transform.scale(ball_image, (ball_radius * 2, ball_radius * 2))

# Symbol images
plus_image = pygame.image.load("assets/green.png").convert_alpha()
plus_image = pygame.transform.scale(plus_image, (ball_radius * 2, ball_radius * 2))

minus_image = pygame.image.load("assets/red.png").convert_alpha()
minus_image = pygame.transform.scale(minus_image, (ball_radius * 2, ball_radius * 2))

infinity_image = pygame.image.load("assets/infinity.png").convert_alpha()
infinity_image = pygame.transform.scale(infinity_image, (ball_radius * 2, ball_radius * 2))

question_image = pygame.image.load("assets/question.png").convert_alpha()
question_image = pygame.transform.scale(question_image, (ball_radius * 2, ball_radius * 2))

# Background image
park_bg = pygame.image.load("assets/park.jpg").convert()
night_bg = pygame.image.load("assets/night.jpg").convert()
sunset_bg = pygame.image.load("assets/sunset.jpg").convert()
blurred_bg = pygame.transform.scale(park_bg, (WIDTH, HEIGHT))

# Load sound effect
bounce_sound = pygame.mixer.Sound("assets/ball.wav")
collect_plus = pygame.mixer.Sound("assets/collect_plus.wav")
collect_minus = pygame.mixer.Sound("assets/collect_minus.wav")
collect_ball = pygame.mixer.Sound("assets/collect_ball.wav")
collect_powerup = pygame.mixer.Sound("assets/collect_powerup.wav")

# Initialize webcam
cap = cv2.VideoCapture(0)

clock = pygame.time.Clock()
running = True

# Score
score = 0
cached_score_text = None
last_score = -1

# Main menu
game_started = False
countdown_active = False
countdown_start_time = 0
countdown_duration = 3

# Hand gesture timer
gesture_hold_start_time = None
gesture_hold_duration = 1.5

# Scaling factor
f = 10.0

# Hand gestures images
five_fingers = pygame.image.load("assets/5.png").convert_alpha()
five_fingers = pygame.transform.scale(five_fingers, (five_fingers.get_width() / f, five_fingers.get_height() / f))

index_finger = pygame.image.load("assets/1.png").convert_alpha()
index_finger = pygame.transform.scale(index_finger, (index_finger.get_width() / (f+2.0), index_finger.get_height() / (f+2.0)))

four_fingers = pygame.image.load("assets/4.png").convert_alpha()
four_fingers = pygame.transform.scale(four_fingers, (four_fingers.get_width() / (f+2.0), four_fingers.get_height() / (f+2.0)))

three_fingers = pygame.image.load("assets/3.png").convert_alpha()
three_fingers = pygame.transform.scale(three_fingers, (three_fingers.get_width() / (f+2.0), three_fingers.get_height() / (f+2.0)))

two_fingers = pygame.image.load("assets/2.png").convert_alpha()
two_fingers = pygame.transform.scale(two_fingers, (two_fingers.get_width() / (f+2.0), two_fingers.get_height() / (f+2.0)))

# Pause flag
game_paused = False
last_toggle_time = 0
cooldown_time = 1.5

# Pause timer
gesture_hold_pause_start_time = None
pause_gesture_duration = 0.25

# Frame settings
frame_counter = 0
process_frequency = 4

# Game over state
game_over = False
selected_option = None

# Controls screen
controls_screen = False
start_menu = True
select_level = False

# Spawn objects
bonus_objects = []
object_spawn_interval = random.randint(3000, 5000)
last_object_spawn_time = pygame.time.get_ticks()

# Power ups
active_powerup = None
powerup_end_time = 0

# Wind mechanic
wind_active = False
wind_start_time = 0
next_wind_time = 0
wind_duration = 1200
wind_vx = 0
wind_vy = 0

# Fullscreen and camera toggle
is_fullscreen = False
camera_on = True

# Font path
font_path = "assets/joystix.otf"

# Check for index finger raised
def index_finger_raised(hand_landmarks):
    fingers = [
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index finger
    ]
    return all(fingers)

# Check for two fingers raised
def two_fingers_raised(hand_landmarks):
    fingers = [
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index finger
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Middle finger
    ]
    return all(fingers)

# Check for three fingers raised
def three_fingers_raised(hand_landmarks):
    fingers = [
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index finger
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Middle finger
        hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Ring finger
    ]
    return all(fingers) and not hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y  # Pinky down

# Check if index, middle, and ring fingers are raised
def four_fingers_raised(hand_landmarks):
    fingers = [
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index finger
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Middle finger
        hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Ring finger
        hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y,  # Pinky finger
    ]
    return all(fingers)

# Check if all five fingers are raised
def all_fingers_raised(hand_landmarks):
    fingers = [
        hand_landmarks.landmark[8].y < hand_landmarks.landmark[6].y,  # Index finger
        hand_landmarks.landmark[12].y < hand_landmarks.landmark[10].y,  # Middle finger
        hand_landmarks.landmark[16].y < hand_landmarks.landmark[14].y,  # Ring finger
        hand_landmarks.landmark[20].y < hand_landmarks.landmark[18].y,  # Pinky finger
        hand_landmarks.landmark[4].x < hand_landmarks.landmark[3].x if hand_landmarks.landmark[4].x > hand_landmarks.landmark[3].x else
        hand_landmarks.landmark[4].y < hand_landmarks.landmark[3].y,  # Thumb
    ]
    return all(fingers)

# Precomputed paddle positions
def discretize_angle(angle, num_bins=90):
    bin_width = 180 / num_bins
    half_bin_width = bin_width / 2
    discrete_angle = round((angle + half_bin_width) / bin_width) * bin_width - half_bin_width
    
    # Neutral range
    neutral_range_start = 0 - half_bin_width
    neutral_range_end = 0 + half_bin_width

    if neutral_range_start <= discrete_angle <= neutral_range_end:
        return 0
    else:
        neutral_midpoint = (neutral_range_start + neutral_range_end) / 2
        return discrete_angle - neutral_midpoint

# Reset the ball (for testing)
def reset_ball():
    global ball_x, ball_y, ball_vx, ball_vy, score
    balls.append({
        "x": ball_x,
        "y": ball_y,
        "vx": ball_vx,
        "vy": ball_vy,
    })
    score = 0

# Reset the paddle
def reset_paddle():
    global smoothed_paddle_x, smoothed_paddle_y, paddle_x, paddle_y
    paddle_x = WIDTH // 2 - paddle_width // 2
    paddle_y = HEIGHT - 50
    smoothed_paddle_x = paddle_x
    smoothed_paddle_y = paddle_y

def f_select_level(background):
    global gesture_hold_start_time, controls_screen, start_menu, select_level, countdown_start_time, blurred_bg
    if gesture_hold_start_time is None:
        gesture_hold_start_time = time.time()
    else:
        elapsed_time = time.time() - gesture_hold_start_time
        if elapsed_time >= gesture_hold_duration:
            controls_screen = False
            start_menu = True
            select_level = False
            countdown_start_time = time.time()
            gesture_hold_start_time = None
            blurred_bg = pygame.transform.scale(background, (WIDTH, HEIGHT))
            return
        
def f_start_menu(next_screen_state):
    global gesture_hold_start_time, game_started, countdown_active, start_menu, select_level, controls_screen, countdown_start_time
    if gesture_hold_start_time is None:
        gesture_hold_start_time = time.time()
    else:
        elapsed_time = time.time() - gesture_hold_start_time
        if elapsed_time >= gesture_hold_duration:
            globals().update(next_screen_state)
            countdown_start_time = time.time()
            gesture_hold_start_time = None
            return
        
def f_game_over(next_screen_state):
    global gesture_hold_start_time, game_started, countdown_active, start_menu, game_over, running, countdown_start_time
    if gesture_hold_start_time is None:
        gesture_hold_start_time = time.time()
    else:
        elapsed_time = time.time() - gesture_hold_start_time
        if elapsed_time >= gesture_hold_duration:
            globals().update(next_screen_state)
            countdown_start_time = time.time()
            gesture_hold_start_time = None
            reset_ball()
            reset_paddle()
            return

def draw_text_and_image(txt, img, height):
    # Font
    font = pygame.font.Font(font_path, 28)

    # Text
    text = font.render(txt, True, WHITE)
    text_rect = text.get_rect(center=(WIDTH // 2 - 20, int(HEIGHT * height)))
    screen.blit(text, text_rect)

    # Image
    gesture_rect = img.get_rect()
    gesture_rect.midleft = (text_rect.right + 20, text_rect.centery)
    screen.blit(img, gesture_rect)

def draw_title(title):
    title_font = pygame.font.Font(font_path, 54)

    title_text = title_font.render(title, True, WHITE)
    title_rect = title_text.get_rect(center=(WIDTH // 2, int(HEIGHT * 0.2)))
    screen.blit(title_text, title_rect)

def draw_background_and_overlay():
    # Background
    screen.blit(blurred_bg, (0, 0))

    # Semi-transparent overlay
    overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    overlay.fill((0, 0, 0, 210))
    screen.blit(overlay, (0, 0))

def spawn_object():
    global bonus_objects
    x = random.randint(50, WIDTH - 50)
    y = random.randint(50, HEIGHT // 2)
    object_type = random.choice(["bonus", "penalty", "multiball", "powerup"])
    # object_type = random.choice(["powerup"])
    bonus_objects.append({"x": x, "y": y, "type": object_type, "radius": 20, "duration": 7000, "object_spawn_time": pygame.time.get_ticks()})

def draw_objects():
    for obj in bonus_objects:
        if obj["type"] == "bonus":
            screen.blit(plus_image, (obj["x"] - obj["radius"], obj["y"] - obj["radius"]))
        elif obj["type"] == "penalty":
            screen.blit(minus_image, (obj["x"] - obj["radius"], obj["y"] - obj["radius"]))
        elif obj["type"] == "powerup":
            screen.blit(question_image, (obj["x"] - obj["radius"], obj["y"] - obj["radius"]))
        else:
            screen.blit(infinity_image, (obj["x"] - obj["radius"], obj["y"] - obj["radius"]))

def check_collisions():
    global score
    for ball in balls:
        for obj in bonus_objects[:]:
            distance = ((ball["x"] - obj["x"]) ** 2 + (ball["y"] - obj["y"]) ** 2) ** 0.5
            if distance < ball_radius + obj["radius"]:
                if obj["type"] == "bonus":
                    score += 5
                    collect_plus.play()
                elif obj["type"] == "penalty":
                    score -= 5
                    collect_minus.play()
                elif obj["type"] == "multiball":
                    spawn_additional_ball(ball["x"], ball["y"])
                    collect_ball.play()
                elif obj["type"] == "powerup":
                    random_powerup()
                    collect_powerup.play()
                bonus_objects.remove(obj)

def spawn_additional_ball(ball_x, ball_y):
    global balls
    balls.append({"x": ball_x, "y": ball_y, "vx": random.choice(np.arange(-1, 1.01, 0.4)), "vy": random.choice(np.arange(-16, -9.99, 2))})

def random_powerup():
    global active_powerup, powerup_end_time
    powerup_choices = ["Double Points", "Shield", "Slow Motion", "Super Bounce"]
    # powerup_choices = ["Shield"]
    active_powerup = random.choice(powerup_choices)
    powerup_end_time = time.time() + 10

def display_powerup():
    global active_powerup
    if active_powerup:
        font = pygame.font.Font(font_path, 28)
        text = font.render(f"{active_powerup}", True, (255, 255, 255))
        text_width = text.get_width()
        screen.blit(text, (screen.get_width() - text_width - 10, 10))

def update_powerups():
    global active_powerup
    if active_powerup and time.time() > powerup_end_time:
        active_powerup = None

def cleanup_objects():
    current_time = pygame.time.get_ticks()
    for obj in bonus_objects[:]:
        if current_time - obj["object_spawn_time"] > obj["duration"]:
            bonus_objects.remove(obj)

def display_wind_status():
    curr_time = pygame.time.get_ticks()
    if curr_time - next_wind_time > -1500 and curr_time - next_wind_time < 0:
        font = pygame.font.Font(font_path, 28)
        text = font.render("WIND INCOMING...", True, (255, 255, 255))
        text_height = text.get_height()
        screen.blit(text, (10, screen.get_height() - text_height - 10))
    elif wind_active:
        font = pygame.font.Font(font_path, 28)
        text = font.render("WIND ACTIVE!", True, (255, 255, 255))
        text_height = text.get_height()
        screen.blit(text, (10, screen.get_height() - text_height - 10))

while running:
    # Capture frame from cam
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process cam frame
    frame = cv2.flip(frame, 1)
    # cv2.imshow("Camera Feed", frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Inside the game loop
    if frame_counter % process_frequency == 0:
        results = hands.process(rgb_frame)
    frame_counter += 1

    if select_level:
        # Detect hands and draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Check if all fingers are raised
                if three_fingers_raised(hand_landmarks):
                    f_select_level(sunset_bg)                    
                elif two_fingers_raised(hand_landmarks):
                    f_select_level(night_bg)
                elif index_finger_raised(hand_landmarks):
                    f_select_level(park_bg)
                else:
                    # Reset the timer if gesture isn't held
                    gesture_hold_start_time = None

        # Background and transparent overlay
        draw_background_and_overlay()

        # Title
        draw_title("SELECT LOCATION")

        # Text and images for gestures
        '''draw_text_and_image("Easy", index_finger, 0.45)
        draw_text_and_image("Medium", two_fingers, 0.6)
        draw_text_and_image("Hard", three_fingers, 0.75)'''
        draw_text_and_image("Park", index_finger, 0.45)
        draw_text_and_image("Court (night)", two_fingers, 0.6)
        draw_text_and_image("Court (sunset)", three_fingers, 0.75)

    if controls_screen and not select_level:
        # Detect hands and draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                )
                
                # Check if all fingers are raised
                if three_fingers_raised(hand_landmarks):
                    if gesture_hold_start_time is None:
                        gesture_hold_start_time = time.time()
                    else:
                        elapsed_time = time.time() - gesture_hold_start_time
                        if elapsed_time >= gesture_hold_duration:
                            controls_screen = False
                            start_menu = True
                            countdown_start_time = time.time()
                            gesture_hold_start_time = None
                            break
                else:
                    # Reset the timer if the gesture isn't held
                    gesture_hold_start_time = None

        # Background and transparent overlay
        draw_background_and_overlay()

        # Title
        draw_title("CONTROLS")

        # Text and images for gestures
        draw_text_and_image("Control Paddle", index_finger, 0.45)
        draw_text_and_image("Pause Game", four_fingers, 0.6)
        draw_text_and_image("Back", three_fingers, 0.75)

    if not game_started and not game_over and start_menu:        
        # Detect hands and draw landmarks on the frame
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp.solutions.drawing_utils.DrawingSpec(color=(0, 255, 0), thickness=2)
                )

                # Check if all fingers are raised
                if all_fingers_raised(hand_landmarks):
                    f_start_menu({'game_started': True, 'countdown_active': True, 'start_menu': False})            
                elif four_fingers_raised(hand_landmarks):
                    f_start_menu({'select_level': True, 'start_menu': False})   
                elif three_fingers_raised(hand_landmarks):
                    f_start_menu({'controls_screen': True, 'start_menu': False})   
                else:
                    # Reset the timer if gesture isn't held
                    gesture_hold_start_time = None

        # Background and transparent overlay
        draw_background_and_overlay()

        # Title
        draw_title("HAND PONG")

        # Text and images for gestures
        draw_text_and_image("Start", five_fingers, 0.45)
        draw_text_and_image("Controls", three_fingers, 0.6)
        draw_text_and_image("Select Location", four_fingers, 0.75)

    if countdown_active:
        # Background and transparent overlay
        draw_background_and_overlay()

        # Elapsed time
        elapsed_time = time.time() - countdown_start_time
        countdown_value = max(0, int(countdown_duration - elapsed_time))

        # Countdown text
        font = pygame.font.Font(None, 120)
        countdown_text = font.render(str(countdown_value), True, WHITE)
        countdown_rect = countdown_text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
        screen.blit(countdown_text, countdown_rect)

        # Next wind time
        next_wind_time = pygame.time.get_ticks() + random.randint(25000, 30000)

        # If countdown is finished, drop the ball
        if elapsed_time >= countdown_duration:
            countdown_active = False
            last_object_spawn_time = pygame.time.get_ticks()
    
    # Check current time
    current_time = time.time()

    # Check for pause gesture
    if game_started and not countdown_active and results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            if four_fingers_raised(hand_landmarks):
                if gesture_hold_pause_start_time is None:
                    gesture_hold_pause_start_time = time.time()
                else:
                    elapsed_time = time.time() - gesture_hold_pause_start_time
                    if elapsed_time >= pause_gesture_duration and current_time - last_toggle_time >= cooldown_time:
                        game_paused = not game_paused
                        last_toggle_time = current_time
                        gesture_hold_pause_start_time = None  # Reset pause timer
            else:
                gesture_hold_pause_start_time = None

    if game_paused:
        # Background and transparent overlay
        draw_background_and_overlay()

        # Display pause text and image
        pause_font = pygame.font.Font(font_path, 54)
        pause_text = pause_font.render("RESUME", True, WHITE)
        pause_text_rect = pause_text.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 50))
        screen.blit(pause_text, pause_text_rect)

        resume_img_rect = four_fingers.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 50))
        screen.blit(four_fingers, resume_img_rect)

    if not game_over and not game_paused and not countdown_active and game_started:
        # Default paddle (hand) position and rotation angle
        paddle_x, paddle_y = smoothed_paddle_x, smoothed_paddle_y
        angle = 0

        # Track hand position and calculate angle
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Camera feed dimensions
                h, w, _ = frame.shape
                index_tip = hand_landmarks.landmark[8]  # Index finger tip
                index_base = hand_landmarks.landmark[5]  # Index finger base

                # Finger position (0.0 to 1.0)
                finger_x_norm = index_tip.x
                finger_y_norm = index_tip.y

                # Map finger position to screen coordinates
                hand_movement_scaling = 1.5
                center_screen_x = WIDTH // 2
                center_screen_y = HEIGHT // 2

                # Paddle position
                scaled_paddle_x = (finger_x_norm - 0.5) * hand_movement_scaling * WIDTH + center_screen_x - paddle_width // 2
                scaled_paddle_y = (finger_y_norm - 0.5) * hand_movement_scaling * HEIGHT + center_screen_y - paddle_height // 2

                # Ensure the paddle stays within bounds
                scaled_paddle_x = max(0, min(WIDTH - paddle_width, scaled_paddle_x))
                scaled_paddle_y = max(0, min(HEIGHT - paddle_height, scaled_paddle_y))

                # Hand angle based on index finger orientation
                dx = index_tip.x - index_base.x
                dy = index_tip.y - index_base.y
                angle = math.degrees(math.atan2(dy, dx)) + 90  # +90 to align with paddle rotation

                # Discretize angle
                angle = discretize_angle(angle, num_bins=90)
                
                # Smooth paddle position for natural movement
                smoothed_paddle_x = paddle_smoothing * smoothed_paddle_x + (1 - paddle_smoothing) * scaled_paddle_x
                smoothed_paddle_y = paddle_smoothing * smoothed_paddle_y + (1 - paddle_smoothing) * scaled_paddle_y

                # Set the smoothed position
                paddle_x, paddle_y = int(smoothed_paddle_x), int(smoothed_paddle_y)

                # Line between the base and tip of the index finger
                base_x = int(index_base.x * w)
                base_y = int(index_base.y * h)
                finger_x_px = int(index_tip.x * w)
                finger_y_px = int(index_tip.y * h)            
                cv2.line(frame, (base_x, base_y), (finger_x_px, finger_y_px), (0, 255, 0), 2)  # Green line

        # End game if no balls left
        if not game_over and len(balls) == 0:
            game_over = True
            gesture_hold_start_time = None
            selected_option = None
            game_started = False
            
        # Update display
        screen.fill((255, 255, 255))

        # Background
        screen.blit(blurred_bg, (0, 0))

        # Paddle surface properties
        paddle_surface = pygame.Surface((paddle_width, paddle_height), pygame.SRCALPHA)
        paddle_surface.fill(BLUE)
        rotated_paddle = pygame.transform.rotate(paddle_image, -angle)
        rotated_rect = rotated_paddle.get_rect(center=(paddle_x + paddle_width // 2, paddle_y + paddle_height // 2))

        # Draw paddle
        screen.blit(rotated_paddle, rotated_rect.topleft)

        # Ball handling
        for ball in balls:
            # Powerups
            if active_powerup == "Slow Motion":
                temp_vy = 0.5
            elif active_powerup == "Super Bounce":
                temp_vy = 1.5
            else:
                temp_vy = 1

            # Wind
            ball["vx"] += wind_vx
            ball["vy"] += wind_vy

            # Update ball position
            ball["x"] += ball["vx"]
            ball["y"] += ball["vy"] * temp_vy
            ball["vy"] += gravity

            # Predictive collision detection with paddle
            paddle_lower_y = paddle_y + 20
            if (abs(angle) > 20):
                paddle_lower_y += 10
            elif (abs(angle) > 10):
                paddle_lower_y += 10

            # Screen edge collisions
            if ball["x"] - ball_radius < 0 or ball["x"] + ball_radius > WIDTH:
                ball["vx"] *= -1

            if (
                paddle_lower_y - ball_radius <= ball["y"] <= paddle_y + paddle_height / 2 and
                paddle_x - ball_radius / 2 < ball["x"] < paddle_x + paddle_width + ball_radius / 2 and
                ball["vy"] > 0
            ):
                # Bounce
                ball["vy"] = bounce_speed

                # Add upward force based on paddle movement
                paddle_vertical_speed = previous_paddle_y - paddle_y
                if paddle_vertical_speed > paddle_speed_threshold:
                    additional_force = paddle_vertical_speed * 1.5
                    ball["vy"] -= additional_force
                
                # Higher bounce heights
                ball["vy"] = max(ball["vy"], bounce_speed * 2)

                # Horizontal velocity based on paddle angle
                ball["vx"] += math.sin(math.radians(angle)) * 10

                # Increase score
                score += 1
                if active_powerup == "Double Points":
                    score += 1

                # Bounce sound
                bounce_sound.play()

            # Draw the ball
            screen.blit(ball_image, (int(ball["x"] - ball_radius), int(ball["y"] - ball_radius)))

            # Shield
            if active_powerup == "Shield":
                if ball["y"] > HEIGHT:
                    ball["x"] = WIDTH // 2
                    ball["y"] = 0
                    ball["vy"] = ball_vy
                    ball["vx"] = 0
                    powerup_end_time = time.time()

                # Semi-transparent surface for the shield
                shield_surface = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
                shield_color = (0, 100, 200, 128)
                pygame.draw.circle(
                    shield_surface, 
                    shield_color, 
                    (int(ball["x"]), int(ball["y"])), 
                    ball_radius + 10
                )
                screen.blit(shield_surface, (0, 0))

            # Ball above screen
            if ball["y"] < 0:
                distance = abs(ball["y"])
                max_distance = HEIGHT
                size_factor = max(10, 40 - int(distance / max_distance * 40)) / 2
                x = ball["x"]
                y = 10

                # Triangle points
                points = [(x, y), (x - size_factor, y + size_factor), (x + size_factor, y + size_factor)]
                pygame.draw.polygon(screen, (255, 255, 255), points)
            
            # Ball out of bounds
            if ball["y"] > HEIGHT:
                balls.remove(ball)

        # Update previous paddle position
        previous_paddle_y = paddle_y

        # Spawn new object after time interval
        current_time = pygame.time.get_ticks()
        if current_time - last_object_spawn_time > object_spawn_interval:
            spawn_object()
            last_object_spawn_time = current_time
        
        check_collisions()
        cleanup_objects()
        draw_objects()

        # Update score display only if score changes
        if score != last_score:
            font = pygame.font.Font(font_path, 28)
            cached_score_text = font.render(f"Score: {score}", True, WHITE)
            last_score = score
        
        # Blit cached score text
        if cached_score_text:
            screen.blit(cached_score_text, (10, 10))

        # Powerups
        display_powerup()
        update_powerups()

        # Wind
        if not wind_active and current_time >= next_wind_time:
            # Activate wind
            wind_active = True
            wind_start_time = current_time
            next_wind_time = current_time + random.randint(25000, 30000)
            wind_vx = random.choice([-0.2, 0.2])
            wind_vy = -0.3

        if wind_active and current_time >= wind_start_time + wind_duration:
            # Deactivate wind after duration
            wind_active = False
            wind_vx = 0
            wind_vy = 0
        
        display_wind_status()

    if game_over:
        # Background and transparent overlay
        draw_background_and_overlay()

        # Title
        draw_title("GAME OVER")

        # Text and images for gestures
        draw_text_and_image("Restart", five_fingers, 0.45)
        draw_text_and_image("Exit to Main Menu", four_fingers, 0.6)
        draw_text_and_image("Quit", three_fingers, 0.75)

        # Gesture detection for options
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Check if all fingers are raised
                if all_fingers_raised(hand_landmarks):
                    f_game_over({'game_started': True, 'countdown_active': True, 'game_over': False})            
                elif four_fingers_raised(hand_landmarks):
                    f_game_over({'start_menu': True, 'game_over': False, 'game_started': False})   
                elif three_fingers_raised(hand_landmarks):
                    f_game_over({'running': False})   
                else:
                    # Reset the timer if gesture isn't held
                    gesture_hold_start_time = None
    
    # Handle camera display logic
    if camera_on:
        frame = cv2.resize(frame, (320, 240))
        
        if not is_fullscreen:
            cv2.imshow("Hand Detection", frame)
        else:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
            frame = cv2.flip(frame, 1)

            frame_surface = pygame.surfarray.make_surface(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_surface = pygame.transform.scale(frame_surface, (176, 132))
            screen.blit(frame_surface, (
                pygame.display.get_surface().get_width() - 176,
                pygame.display.get_surface().get_height() - 132
            ))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_f:  # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    pygame.display.set_mode((100, 0), pygame.FULLSCREEN)
                else:
                    pygame.display.set_mode((800, 600))
            elif event.key == pygame.K_c:  # Toggle camera
                camera_on = not camera_on
                # print(camera_on)
                if not camera_on and not is_fullscreen:
                    cv2.destroyWindow("Hand Detection")
            elif event.key == pygame.K_q:
                running = False
    
    # Show camera feed
    # frame = cv2.resize(frame, (320, 240))
    # cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        running = False

    # Update display
    pygame.display.flip()
    clock.tick(60)

    # print(f"FPS: {clock.get_fps()}")

# Cleanup
cap.release()
cv2.destroyAllWindows()
pygame.quit()