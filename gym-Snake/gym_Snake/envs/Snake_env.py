import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math
import pygame
from pygame import gfxdraw
from pygame.locals import *
import os
import threading
import time

# Objects
EMPTY = 0
WALL = 1
TARGET = 2
BODY = 3
DIGESTION = 4
HEAD_UP = 5
HEAD_RIGHT = 6
HEAD_DOWN = 7
HEAD_LEFT = 8

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Rewards
COLLISION_REWARD = -10
TARGET_REWARD = 1
SURVIVED_REWARD = 0

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'print'], "render_fps": 50}


    def __init__(self, width=10, height=10, solid_border=True, mode='computer'):
        # Store informations
        self.__board_width = width
        self.__board_height = height
        self.__board_solid_border = solid_border
        # Ensure actions are valid
        self.__possible_actions = [0, 1, 2]
        # Initialize variables
        self.reset(mode=mode)



    # Actions: 0 continue, 1 turn right, 2 turn left
    def step(self, action):
        if action not in self.__possible_actions:
            action = 0
        # Get new direction
        if action == 0:
            pass
        if action == 1:
            self.__direction = (self.__direction + 1) % 4
        if action == 2:
            self.__direction = (self.__direction + 3) % 4
        # Get new head position
        old_h, old_w = self.__head_pos
        if self.__direction == UP:
            new_h_pos = (old_h - 1, old_w)
        if self.__direction == RIGHT:
            new_h_pos = (old_h, old_w + 1)
        if self.__direction == DOWN:
            new_h_pos = (old_h + 1, old_w)
        if self.__direction == LEFT:
            new_h_pos = (old_h, old_w - 1)
        # Check if collision
        if self.__board[new_h_pos] == WALL or self.__board[new_h_pos] == BODY:
            return self.__board.copy(), COLLISION_REWARD, True, {}
        # Check if target
        if self.__board[new_h_pos] == TARGET:
            reward = TARGET_REWARD
            self.__digestion.append(len(self.__snake_path) + len(self.__digestion) + 1) # Digest when the last part of the tail reaches the target position
            # Place a new target
            self.__place_target()
        else:
            reward = SURVIVED_REWARD
        # Move
        self.__move_snake(self.__direction)
        # Update board
        self.__remove_snake()
        self.__place_snake()
        # Digest by one all the target eated
        self.__digestion = [x - 1 for x in self.__digestion]
        # Return
        return self.__board.copy(), reward, False, {}


    def reset(self, width=None, height=None, solid_border=None, mode='computer'):
        # Possibly change environment
        if width is not None: self.__board_width = width
        if height is not None: self.__board_height = height
        if solid_border is not None: self.__board_solid_border = solid_border
        # Define action space
        # 3 possible actions: continue in the same direction, turn right, turn left
        self.action_space = spaces.Discrete(3)
        # Define observation space
        self.observation_space = spaces.Discrete(self.__board_width * self.__board_height)
        # Keep track of targets to digest
        self.__digestion = []
        # Generate the board
        self.__generate_board()
        # Generate the snake
        self.__generate_snake()
        # Place the snake on the board
        self.__place_snake()
        # Place the first target
        self.__place_target()
        # Screen for render
        self.__screen = None
        # Clock for render
        self.__clock = None
        # Screen width
        self.__screen_width = 600
        # Screen height
        self.__screen_height = 400
        # Check game mode
        self.__play_mode = mode
        if mode == 'human':
            self.__play_human()


    # Generate the board on which the snake moves
    def __generate_board(self):
        # Generate empty matrix
        b = np.zeros((self.__board_height, self.__board_width), dtype=int)
        # If necessary, create the border
        if self.__board_solid_border:
            # Create walls on the first row
            b[0,:] = WALL
            # Create walls on the last row
            b[self.__board_height - 1,:] = WALL
            # Create walls on the first column
            b[:,0] = WALL
            # Create walls on the last column
            b[:, self.__board_width -1 ] = WALL
        # Set as board
        self.__board = b


    # Generate the snake on the board
    # Make sure the tile in front of the snake is free
    # TODO: make more stable (to different envs shape)
    def __generate_snake(self):
        # Get coordinates of the central tiles
        w = math.ceil(self.__board_width / 2) - 1
        h = math.ceil(self.__board_height / 2) - 1
        # Place head
        self.__head_pos = [h, w]
        # Store direction -> 0 for UP, 1 for RIGHT, 2 for DOWN, 3 for LEFT
        self.__direction = LEFT
        # Create snake (only 1 tile on the right of the head)
        # self.__snake_path = [RIGHT, RIGHT, UP, RIGHT, DOWN, DOWN, DOWN, LEFT]
        self.__snake_path = [RIGHT]


    # Remove the snake from the board (before re-placing it)
    def __remove_snake(self):
        # Remove head
        self.__board[self.__head_pos] = EMPTY
        # Remove body
        self.__board[self.__board == BODY] = EMPTY


    # Place the snake on the board, based on the snake_path
    def __place_snake(self):
        # Place the head (and show the direction)
        self.__board[self.__head_pos[0], self.__head_pos[1]] = HEAD_UP + self.__direction
        # Store coordinates
        temp_h, temp_w = self.__head_pos
        # Keep track of digestion
        d = len(self.__snake_path)
        # Add body
        for direction in self.__snake_path:
            if direction == UP:
                temp_h -= 1
            if direction == RIGHT:
                temp_w += 1
            if direction == DOWN:
                temp_h += 1
            if direction == LEFT:
                temp_w -= 1
            # Add body
            if d in self.__digestion:
                self.__board[temp_h, temp_w] = DIGESTION
            else:
                self.__board[temp_h, temp_w] = BODY
            d -= 1


    # TODO verify digestion
    # Move the snake by 1 tile
    def __move_snake(self, direction, eating=False):
        # Save position of the head
        old_h, old_w = self.__head_pos
        # Move the head
        if direction == UP:
            self.__head_pos = (old_h - 1, old_w)
        if direction == RIGHT:
            self.__head_pos = (old_h, old_w + 1)
        if direction == DOWN:
            self.__head_pos = (old_h + 1, old_w)
        if direction == LEFT:
            self.__head_pos = (old_h, old_w - 1)
        # Move body
        # Insert a body tile behind the head
        self.__snake_path.insert(0, (direction + 2) % 4)
        # If necessary, remove last part
        if 0 not in self.__digestion:
            self.__snake_path.pop()
        # Else, finish digestion
        else:
            self.__digestion.remove(0)


    # Place the target
    # Randomly choose one of the empty slots
    def __place_target(self):
        # Get empty tiles
        possible = np.where(self.__board == EMPTY)
        # If any empty tile found, place a target
        if len(possible) >= 0:
            pos = np.random.randint(len(possible[0]))
            self.__board[possible[0][pos], possible[1][pos]] = TARGET


    # TODO add comments on what each line does (for pygame code)
    def render(self, mode='human', close=False):
        if mode == 'print':
            print(self.__board)
            # print(self.__digestion)
        elif mode == 'human':
            # If necessary, init window
            if self.__screen is None:
                pygame.init()
                pygame.display.init()
                # Set window title
                pygame.display.set_caption('Snake')
                # Set window icon
                # Move working directory
                try:
                    abspath = os.path.abspath(__file__)
                    dname = os.path.dirname(abspath)
                    os.chdir(dname)
                    Icon = pygame.image.load('../../data/snake.png')
                    pygame.display.set_icon(Icon)
                except Exception as e:
                    pass
                # Set screen size
                self.__screen = pygame.display.set_mode((self.__screen_width, self.__screen_height))
                # Compute the necessary measures
                self.__compute_measures()
                # Check if the close button in pressed
                # th = threading.Thread(target=__check_closing, args=(self))
                if self.__play_mode != 'human':
                    th = threading.Thread(target=self.__check_closing, args=[pygame])
                    th.start()
            if self.__clock is None:
                self.__clock = pygame.time.Clock()
            # Create painting surface
            self.__surf = pygame.Surface((self.__screen_width, self.__screen_height))
            # Fill surface with white
            self.__surf.fill(self.__bg_color)
            # Draw cells
            for y in range(len(self.__board)):
                for x in range(len(self.__board[y])):
                    x_coord = self.__tile_width * x + self.__margin_left
                    y_coord = self.__tile_height * y + self.__margin_top
                    color = self.__colors[self.__board[self.__board_height - 1 - y,x]]
                    # Draw
                    gfxdraw.box(self.__surf, pygame.Rect(x_coord, y_coord, self.__tile_width, self.__tile_height), color) # pygame y coordinates uses 0 on top and grows toward bottom, therefore need to invert y

            # Render
            self.__surf = pygame.transform.flip(self.__surf, False, True)
            self.__screen.blit(self.__surf, (0, 0))
            # pygame.event.pump()
            self.__clock.tick(self.metadata["render_fps"])
            pygame.display.flip()


    def __check_closing(self, pg):
        # time.sleep(2)
        # pg.init()
        self.__running = True
        while self.__running:
            pg.init()
            for event in pg.event.get():
                if event.type == QUIT:
                    self.close()
                    self.__screen = None
                    self.__clock = None
                    self.__running = False


    # Compute useful variables for the graphics
    def __compute_measures(self):
        # Use at least 10% of margin
        self.__margin_left = int(self.__screen_width / 20)
        self.__margin_top = int(self.__screen_height / 20)
        # Compute the size of each cell
        max_size = int(min((self.__screen_width - self.__margin_left) / self.__board_width, (self.__screen_height - self.__margin_top) / self.__board_height))
        # Correct margins
        self.__margin_left = int((self.__screen_width - max_size * self.__board_width) / 2)
        self.__margin_top = int((self.__screen_height - max_size * self.__board_height) / 2)
        # Ensure the cells are squared
        self.__tile_width = max_size
        self.__tile_height = max_size
        # Colors for each type of cell (from 0 to 7)
        empty_color = (255, 255, 255)
        wall_color = (0, 0, 0)
        target_color = (255, 0, 0)
        body_color = (0, 255, 0)
        digestion_color = (200, 200, 0)
        head_up_color = (0, 128, 0)
        head_right_color = (0, 128, 0)
        head_down_color = (0, 128, 0)
        head_left_color = (0, 128, 0)
        self.__colors = [empty_color, wall_color, target_color, body_color, digestion_color, head_up_color, head_right_color, head_down_color, head_left_color]
        # Colors for the background
        self.__bg_color = (25, 250, 250)


    def close(self):
        if self.__screen is not None:
            pygame.display.quit()
            pygame.quit()


    # Play in human mode. Useful to debug
    def __play_human(self):
        self.__running = True
        done = False

        # Time to do a move
        self.__timeout = 1000 # In ms

        while self.__running and not done:
            # Render
            # Sometime error when closing
            try:
                self.render()
            except Exception as e:
                pass


            # Limit time
            current_time = pygame.time.get_ticks()
            next_step = current_time + int(self.__timeout)

            # Default move
            next_move = 0

            # Check input
            while current_time < next_step and self.__running:
                # pygame.init()
                current_time = pygame.time.get_ticks()
                for event in pygame.event.get():
                    # If a key is pressed
                    if event.type == QUIT:
                        self.__running = False
                    if event.type == KEYDOWN:
                        # Action depends on current direction as well
                        if self.__direction == UP:
                            # if event.key == ord("w") or event.key == pygame.K_UP:
                            #     print("You pressed w or key up")
                            if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                                # print("You pressed d or key right")
                                next_move = RIGHT
                            # elif event.key == ord( "s" ) or event.key == pygame.K_DOWN:
                            #     print("You pressed s or key down")
                            elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                                # print("You pressed a or key left")
                                next_move = 2
                        elif self.__direction == RIGHT:
                            if event.key == pygame.K_w or event.key == pygame.K_UP:
                                # print("You pressed w or key up")
                                next_move = 2
                            # elif event.key == ord( "d" ) or event.key == pygame.K_RIGHT:
                            #     print("You pressed d or key right")
                            elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                                # print("You pressed s or key down")
                                next_move = 1
                            # elif event.key == ord( "a" ) or event.key == pygame.K_LEFT:
                            #     print("You pressed a or key left")
                        elif self.__direction == DOWN:
                            # if event.key == ord("w") or event.key == pygame.K_UP:
                            #     print("You pressed w or key up")
                            if event.key == pygame.K_d or event.key == pygame.K_RIGHT:
                                next_move = 2
                            #     print("You pressed d or key right")
                            # elif event.key == ord( "s" ) or event.key == pygame.K_DOWN:
                            #     print("You pressed s or key down")
                            elif event.key == pygame.K_a or event.key == pygame.K_LEFT:
                                next_move = 1
                                # print("You pressed a or key left")
                        elif self.__direction == LEFT:
                            if event.key == pygame.K_w or event.key == pygame.K_UP:
                                # print("You pressed w or key up")
                                next_move = 1
                            # elif event.key == ord( "d" ) or event.key == pygame.K_RIGHT:
                            #     print("You pressed d or key right")
                            elif event.key == pygame.K_s or event.key == pygame.K_DOWN:
                            #     print("You pressed s or key down")
                            # elif event.key == ord( "a" ) or event.key == pygame.K_LEFT:
                            #     print("You pressed a or key left")
                                next_move = 2

            # Move
            _, reward, done, _ = self.step(next_move)

            # Reduce timeout
            if reward == TARGET_REWARD:
                self.__timeout *= 0.95

            # Reset default move
            next_move = 0

        self.close()
        self.__screen = None
        self.__clock = None
