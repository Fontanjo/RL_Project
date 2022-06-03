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

HEAD_BLOCKS = [HEAD_UP, HEAD_RIGHT, HEAD_DOWN, HEAD_LEFT]
COLLISION_BLOCKS = [WALL, BODY, DIGESTION]

# Directions
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3

# Rewards
REWARD_TARGET = 10
REWARD_COLLISION = -100
REWARD_TOWARD = 1
REWARD_AWAY = -1
REWARD_SURVIVED = 0

# Human mode
INITIAL_TIMEOUT = 1000 # In ms
TIMEOUT_DECREASE = 0.95

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human', 'print'], 'state_mode': ['matrix', 'states'], 'reward_mode': ['normal', 'extended','adaptive'], "render_fps": 50}


    def __init__(self, seed=None, width=10, height=10, solid_border=True, shape='Normal', custom_board=None, player='computer', state_mode='states', reward_mode='normal', rewards=None):
        # Set seed for random
        if seed is not None: np.random.seed(seed)
        # Import global variables to possibly modify them
        global REWARD_TARGET, REWARD_COLLISION, REWARD_TOWARD, REWARD_AWAY, REWARD_SURVIVED
        # Store informations
        if custom_board != None:
            # Remember that there is a customized board
            self.__board_type = 'Custom'
            # Initialize custom board
            self.__generate_custom_board(custom_board)
        else:
            self.__board_width = width
            self.__board_height = height
            self.__board_solid_border = solid_border
            self.__board_type = shape
        # Save reward mode
        self.__reward_mode = reward_mode
        # Change rewards
        if rewards is not None:
            REWARD_TARGET       = rewards.get('REWARD_TARGET', REWARD_TARGET)
            REWARD_COLLISION    = rewards.get('REWARD_COLLISION', REWARD_COLLISION)
            REWARD_TOWARD       = rewards.get('REWARD_TOWARD', REWARD_TOWARD)
            REWARD_AWAY         = rewards.get('REWARD_AWAY', REWARD_AWAY)
            REWARD_SURVIVED     = rewards.get('REWARD_SURVIVED', REWARD_SURVIVED)
        # Save state mode
        self.__state_mode = state_mode
        # Ensure actions are valid
        self.__possible_actions = [0, 1, 2]
        # Initialize variables
        self.reset(player=player)



    # Actions: 0 continue, 1 turn right, 2 turn left
    def step(self, action):
        # Update step count
        self.__total_steps += 1
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
            new_h_pos = ((old_h + self.__board_height - 1) % self.__board_height, old_w)
        if self.__direction == RIGHT:
            new_h_pos = (old_h, (old_w + 1) % self.__board_width)
        if self.__direction == DOWN:
            new_h_pos = ((old_h + 1) % self.__board_height, old_w)
        if self.__direction == LEFT:
            new_h_pos = (old_h, (old_w + self.__board_width - 1) % self.__board_width)

        # Check if done
        done = False

        # Check if collision
        if self.__board[new_h_pos] in COLLISION_BLOCKS:
            rew = self.__compute_reward(True, False)
            done = True
        # Check if target
        elif self.__board[new_h_pos] == TARGET:
            self.__digestion.append(len(self.__snake_path) + 1) # Digest when the last part of the tail reaches the target position
            # Place a new target
            self.__place_target()
            rew = self.__compute_reward(False, True)
        else:
            rew = self.__compute_reward(False, False, old_pos = (old_h, old_w), new_pos = new_h_pos)
        # Move
        self.__move_snake(self.__direction)
        # Update board
        self.__remove_snake()
        self.__place_snake()
        # Digest by one all the target eated
        self.__digestion = [x - 1 for x in self.__digestion]
        # Compute new state and info
        s, info = self.__compute_state_info()
        # Update total reward
        self.__total_reward += rew
        # Return
        return s, rew, done, info


    def reset(self, seed=None, width=None, height=None, solid_border=None, shape=None, custom_board=None, player='computer', state_mode=None, reward_mode=None, rewards=None):
        # Set seed for random
        if seed is not None: np.random.seed(seed)
        # Import global variables to possibly modify them
        global REWARD_TARGET, REWARD_COLLISION, REWARD_TOWARD, REWARD_AWAY, REWARD_SURVIVED
        # If a new custom board is added, or if there was one before and no new shape has been specified
        if custom_board != None or (shape == None and self.__board_type == 'Custom'):
            # Remember that there is a customized board
            self.__board_type = 'Custom'
            # Get new board, or old one if new one not specified
            b = custom_board
            if b == None:
                b = self.__board_base
            # Initialize custom board
            self.__generate_custom_board(custom_board)
        else:
            # Possibly change environment
            if width is not None: self.__board_width = width
            if height is not None: self.__board_height = height
            if solid_border is not None: self.__board_solid_border = solid_border
            if shape is not None: self.__board_type = shape
            # Generate the board
            self.__generate_board()
        # Possibly change reward mode
        if reward_mode is not None: self.__reward_mode = reward_mode
        # Possibly change state mode
        if state_mode is not None: self.__state_mode = state_mode
        # Define action space
        # 3 possible actions: continue in the same direction, turn right, turn left
        self.action_space = spaces.Discrete(3)
        # Define observation space
        if self.__state_mode == 'matrix':
            self.observation_space = spaces.Discrete(self.__board_width * self.__board_height)
        else:
            self.observation_space = spaces.Discrete(2**10)
        # Keep track of targets to digest
        self.__digestion = []
        # Keep track of steps
        self.__total_steps = 0
        # Keep track of the total reward
        self.__total_reward = 0
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
        # Info table width
        self.__info_table_width = 300
        # Screen height
        self.__screen_height = 700
        # Check game player
        self.__player = player
        if self.__player == 'human':
            self.__play_human()
        # Return initial state
        s, _ = self.__compute_state_info()
        return s


    def __compute_reward(self, done, target, old_pos = None, new_pos = None):
        # Compute reward
        if done:
            r = REWARD_COLLISION
        elif target:
            r = REWARD_TARGET
        else:
            r = REWARD_SURVIVED
            if self.__reward_mode in ['extended', 'adaptive']:
                # Get target position
                target_x, target_y = self.__target_position
                # Get head positions
                assert old_pos is not None
                assert new_pos is not None
                old_x, old_y = old_pos
                new_x, new_y = new_pos
                # Check if we got closer or away
                dist_before = np.abs(target_x - old_x) + np.abs(target_y - old_y)
                dist_after = np.abs(target_x - new_x) + np.abs(target_y - new_y)
                # Possibly adapt reward according to body length
                divider = 1
                if self.__reward_mode == 'adaptive':
                    divider = len(self.__snake_path) # + 1 # Add 1 to include head
                # Ev add reward
                if dist_before > dist_after:
                    r += REWARD_TOWARD / divider
                elif dist_before < dist_after:
                    r += REWARD_AWAY / divider

        return r


    # Compute reward based on old state, new state (action), and state mode
    def __compute_state_info(self):
        # Compute info
        i = {}


        # Compute new state
        if self.__state_mode == 'matrix':
            # State
            s = self.__board.copy()
        elif self.__state_mode == 'states':
            # In state mode, the state is represented by a single int
            # The int is the decimal representation of a binary array defined as following:
            # 0-2 bits: target position (8 possibilities)
            # 3-9 bits: obstacle in front or not, obstacle in front-right or not, obstacle on the right or not, ...
                # Obstacle behind always true, so ignore
            bit_str = ''

            # Normalize matrix (make snake face up)
            if self.__direction == UP:
                m = self.__board.copy()
                hx, hy = self.__head_pos
                tx, ty = self.__target_position
            elif self.__direction == RIGHT:
                m = self.__rotate_matrix_counter_clockwise(self.__board.copy())
                # Get coordinates on new system
                hx, hy = np.where(np.array(m) == HEAD_RIGHT)
            elif self.__direction == DOWN:
                m = self.__rotate_matrix_counter_clockwise(self.__board.copy())
                m = self.__rotate_matrix_counter_clockwise(m)
                # Get coordinates on new system
                hx, hy = np.where(np.array(m) == HEAD_DOWN)
            elif self.__direction == LEFT:
                m = self.__rotate_matrix_counter_clockwise(self.__board.copy())
                m = self.__rotate_matrix_counter_clockwise(m)
                m = self.__rotate_matrix_counter_clockwise(m)
                # Get head coordinates on new system
                hx, hy = np.where(np.array(m) == HEAD_LEFT)

            # Get target coordinates on new system
            tx, ty = np.where(np.array(m) == TARGET)
            # Security checks
            assert len(tx) == 1
            assert len(ty) == 1
            if self.__direction != UP:
                assert len(hx) == 1, f"h_pos = {self.__head_pos}\nboard = {self.__board}\nm = {np.array(m)}"
                assert len(hy) == 1
            # np.where returns an array, extract first (and only) element
            tx, ty = tx[0], ty[0]
            if self.__direction != UP:
                hx, hy = hx[0], hy[0]


            # Target position
            target_pos_matrix = np.zeros((3,3))
            # tx, ty = self.__target_position
            # hx, hy = self.__head_pos
            if tx < hx and ty == hy:
                bit_str += '000'
                target_pos_matrix[0, 1] = 1
            elif tx < hx and ty > hy:
                bit_str += '001'
                target_pos_matrix[0, 2] = 1
            elif tx == hx and ty > hy:
                bit_str += '010'
                target_pos_matrix[1, 2] = 1
            elif tx > hx and ty > hy:
                bit_str += '011'
                target_pos_matrix[2, 2] = 1
            elif tx > hx and ty == hy:
                bit_str += '100'
                target_pos_matrix[2, 1] = 1
            elif tx > hx and ty < hy:
                bit_str += '101'
                target_pos_matrix[2, 0] = 1
            elif tx == hx and ty < hy:
                bit_str += '110'
                target_pos_matrix[1, 0] = 1
            elif tx > hx and ty < hy:
                bit_str += '111'
                target_pos_matrix[0, 0] = 1

            # Add info
            i['target_pos'] = target_pos_matrix

            # Convert to np array to use better indices
            m = np.array(m)

            # Obstacles
            walls_matrix = np.zeros((3,3))
            h, w = len(m), len(m[0])
            # Obstacle in front
            if m[(hx - 1 + h) % h, hy] in COLLISION_BLOCKS:
                bit_str += '1'
                walls_matrix[0,1] = 1
            else:
                bit_str += '0'
            # Obstacle front right
            if m[(hx - 1 + h) % h, (hy + 1) % w] in COLLISION_BLOCKS:
                bit_str += '1'
                walls_matrix[0,2] = 1
            else:
                bit_str += '0'
            # Obstacle right
            if m[hx, (hy + 1) % w] in COLLISION_BLOCKS:
                bit_str += '1'
                walls_matrix[1,2] = 1
            else:
                bit_str += '0'
            # Obstacle behind right
            if m[(hx + 1) % h, (hy + 1) % w] in COLLISION_BLOCKS:
                bit_str += '1'
                walls_matrix[2,2] = 1
            else:
                bit_str += '0'
            # Behind the head there is always a block (body)
            walls_matrix[2,1] = 1
            # Obstacle behind left
            if m[(hx + 1) % h, (hy - 1 + w) % w] in COLLISION_BLOCKS:
                bit_str += '1'
                walls_matrix[2,0] = 1
            else:
                bit_str += '0'
            # Obstacle left
            if m[hx, (hy - 1 + w) % w] in COLLISION_BLOCKS:
                bit_str += '1'
                walls_matrix[1,0] = 1
            else:
                bit_str += '0'
            # Obstacle front left
            if m[(hx - 1 + h) % h, (hy - 1 + w) % w] in COLLISION_BLOCKS:
                bit_str += '1'
                walls_matrix[0,0] = 1
            else:
                bit_str += '0'


            # Add info
            i['walls'] = walls_matrix

            # Convert binary string to int
            s = int(bit_str, 2)

        # Return new state, reward, info
        return s, i


    # Return a matrix that is the counter clockwise 90 degree rotation of the input matrix
    def __rotate_matrix_counter_clockwise(self, m):
        return [[m[j][i] for j in range(len(m))] for i in range(len(m[0])-1,-1,-1)]

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
        # If necessary, add new walls
        if self.__board_type == 'Shuriken':
            self.__generate_shuriken_board()
        elif self.__board_type == 'Double_h':
            self.__generate_double_h_board()
        elif self.__board_type == 'Double_v':
            self.__generate_double_v_board()
        elif self.__board_type == 'Maze':
            print('Maze board not yet implemented')
        # Keep shape of empty board
        self.__board_base = self.__board.copy()


    # Generate custom board
    # TODO: ensure there are only 1s and 0s (or handle the case where the snake and/or the target are already given)
    def __generate_custom_board(self, custom_board):
        # Keep shape of empty board (for reset)
        self.__board_base = custom_board.copy()
        # Cast to numpy array
        b = np.array(custom_board)
        # Ensure it is 2D
        assert len(b.shape) == 2, 'The board should be a 2D matrix'
        # Get height
        self.__board_height = b.shape[0]
        # Get width
        self.__board_width = b.shape[1]
        # Save board
        self.__board = b.copy()


    # Generate shuriken board
    def __generate_shuriken_board(self):
        # Make sure the minimal dimension are respected
        if self.__board_solid_border:
            assert self.__board_width >= 8 and self.__board_height >= 8, 'For Shuriken shape with solid border, the board should be at least 8x8'
        else:
            assert self.__board_width >= 6 and self.__board_height >= 6, 'For Shuriken shape (without solid border), the board should be at least 6x6'
        # Add walls
        # Number of upper and lower walls. Remove at least 4 for center. First one overrider border if present
        vertical_nb = int((self.__board_height - 6) / 2) + 1
        # Upper wall index
        up_w_index = int(self.__board_width / 2)
        # Add upper walls
        for h in range(vertical_nb):
            self.__board[h, up_w_index] = WALL
        # Lower wall index
        down_w_index = int((self.__board_width + 1) / 2) - 1
        # Add lower walls
        for h in range(vertical_nb):
            self.__board[self.__board_height - 1 - h, down_w_index] = WALL
        # Number of right and left walls. Remove 2 for borders, and at least 4 for center
        horizontal_nb = int((self.__board_width - 6) / 2) + 1
        # Left wall index
        left_h_index = int((self.__board_height + 1) / 2) - 1
        # Add left walls
        for w in range(horizontal_nb):
            self.__board[left_h_index, w] = WALL
        # Right wall index
        right_h_index = int(self.__board_height / 2)
        # Add right walls
        for w in range(horizontal_nb):
            self.__board[right_h_index, self.__board_width - 1 - w] = WALL


    # Generate double board (horizontal separation)
    # TODO add check for minimal size
    def __generate_double_h_board(self):
        # Upper wall index
        left_h_index = int(self.__board_height / 2)
        # Number of upper walls. Remove at least 1 for center. First one overrider border if present
        left_nb = int((self.__board_width - 1) / 2)
        # Add upper walls
        for w in range(left_nb):
            self.__board[left_h_index, w] = WALL
        # Lower wall index
        right_w_index = int((self.__board_height + 1) / 2) - 1
        # Number of lower walls. Remove at least 1 for center. First one overrider border if present
        right_nb = int((self.__board_width - 1) / 2) + 1
        # Add lower walls
        for w in range(right_nb):
            self.__board[right_w_index, self.__board_width - 1 - w] = WALL


    # Generate double board (vertical separation)
    # TODO add check for minimal size
    def __generate_double_v_board(self):
        # Upper wall index
        up_w_index = int(self.__board_width / 2)
        # Number of upper walls. Remove at least 1 for center. First one overrider border if present
        up_nb = int((self.__board_height - 1) / 2)
        # Add upper walls
        for h in range(up_nb):
            self.__board[h, up_w_index] = WALL
        # Lower wall index
        down_w_index = int((self.__board_width + 1) / 2) - 1
        # Number of lower walls. Remove at least 1 for center. First one overrider border if present
        down_nb = int((self.__board_height - 1) / 2) + 1
        # Add lower walls
        for h in range(down_nb):
            self.__board[self.__board_height - 1 - h, down_w_index] = WALL


    # Generate the snake on the board
    # TODO: make sure the tile in front of the snake is free
    # TODO: make more stable (to different envs shape)
    def __generate_snake(self):
        # Get coordinates of the central tiles
        w = math.ceil(self.__board_width / 2) - 1
        h = math.ceil(self.__board_height / 2) - 1
        # Place head
        self.__head_pos = [h, w]
        # Create snake (only 1 tile on the right of the head)
        # self.__snake_path = [RIGHT, RIGHT, UP, RIGHT, DOWN, DOWN, DOWN, LEFT]
        if self.__board[h, w+1] == EMPTY:
            # Store direction -> 0 for UP, 1 for RIGHT, 2 for DOWN, 3 for LEFT
            self.__direction = LEFT
            self.__snake_path = [RIGHT]
        else:
            self.__direction = DOWN
            self.__snake_path = [UP]

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
            elif direction == RIGHT:
                temp_w += 1
            elif direction == DOWN:
                temp_h += 1
            elif direction == LEFT:
                temp_w -= 1
            # Pacman effect
            temp_w = (temp_w + self.__board_width) % self.__board_width
            temp_h = (temp_h + self.__board_height) % self.__board_height
            # Add body
            if d in self.__digestion:
                # Do not override head (when die)
                if temp_h != self.__head_pos[0] or temp_w != self.__head_pos[1]:
                    self.__board[temp_h, temp_w] = DIGESTION
            else:
                # Do not override head (when die)
                if temp_h != self.__head_pos[0] or temp_w != self.__head_pos[1]:
                    self.__board[temp_h, temp_w] = BODY
            d -= 1


    # Move the snake by 1 tile
    def __move_snake(self, direction, eating=False):
        # Save position of the head
        old_h, old_w = self.__head_pos
        # Move the head
        if direction == UP:
            self.__head_pos = ((old_h + self.__board_height - 1) % self.__board_height, old_w)
        if direction == RIGHT:
            self.__head_pos = (old_h, (old_w + 1) % self.__board_width)
        if direction == DOWN:
            self.__head_pos = ((old_h + 1) % self.__board_height, old_w)
        if direction == LEFT:
            self.__head_pos = (old_h, (old_w + self.__board_width - 1) % self.__board_width)
        # Move body
        # Insert a body tile behind the head
        self.__snake_path.insert(0, (direction + 2) % 4)
        # If necessary, remove last part
        if 0 not in self.__digestion:
            self.__snake_path.pop()
        # Else, finish digestion
        else:
            self.__digestion.remove(0)
            # Update remaining digestions
            self.__digestion = [x + 1 for x in self.__digestion]


    # Place the target
    # Randomly choose one of the empty slots
    def __place_target(self):
        # Get empty tiles
        possible = np.where(self.__board == EMPTY)
        # If any empty tile found, place a target
        if len(possible[0]) > 0:
            pos = np.random.randint(len(possible[0]))
            self.__board[possible[0][pos], possible[1][pos]] = TARGET
            self.__target_position = [possible[0][pos], possible[1][pos]]
        else:
            print('Game finished! You won!')


    # TODO add comments on what each line does (for pygame code)
    def render(self, mode='human', close=False):
        if mode == 'print':
            print(self.__board)
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
                self.__screen = pygame.display.set_mode((self.__screen_width + self.__info_table_width, self.__screen_height))
                # Compute the necessary measures
                self.__compute_measures()
                # Check if the close button in pressed
                # th = threading.Thread(target=__check_closing, args=(self))
                if self.__player != 'human':
                    th = threading.Thread(target=self.__check_closing, args=[pygame])
                    th.start()
            if self.__clock is None:
                self.__clock = pygame.time.Clock()
            # Create painting surface
            self.__surf = pygame.Surface((self.__screen_width + self.__info_table_width, self.__screen_height))
            # Fill surface with white
            self.__surf.fill(self.__bg_color)
            # Draw cells
            for y in range(len(self.__board)):
                for x in range(len(self.__board[y])):
                    x_coord = self.__tile_width * x + self.__margin_left
                    y_coord = self.__tile_height * y + self.__margin_top
                    color = self.__colors[self.__board[self.__board_height - 1 - y, x]] # pygame y coordinates uses 0 on top and grows toward bottom, therefore need to invert y
                    # Draw
                    gfxdraw.box(self.__surf, pygame.Rect(x_coord, y_coord, self.__tile_width, self.__tile_height), color)
            # Draw decorations
            self.__draw_decorations()
            # # Draw information table
            self.__draw_info_table()
            # Render
            self.__surf = pygame.transform.flip(self.__surf, False, True)
            self.__screen.blit(self.__surf, (0, 0))
            # pygame.event.pump()
            self.__clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
            # self.__screen.update()


    # Draw decorations on the snake
    def __draw_decorations(self):
        # Get coordinates of head
        hh, hw = self.__head_pos
        hh = self.__board_height - 1 - hh  # Since matrix and pygame coord system are opposed
        x_coord_head = self.__tile_width * hw + self.__margin_left
        y_coord_head = self.__tile_height * hh + self.__margin_top
        # Distinguish head direction
        if self.__direction == UP:
            # Delete first quarter to make head smaller (and make space for tongue)
            gfxdraw.box(self.__surf, pygame.Rect(x_coord_head, y_coord_head + self.__tile_height, self.__tile_width, -(self.__tile_height / 4)), self.__colors[EMPTY])
            # Draw tongue
            gfxdraw.box(self.__surf, pygame.Rect(x_coord_head + (self.__tile_width * 3 / 8), y_coord_head + self.__tile_height, self.__tile_width / 4, -(self.__tile_height / 4)), self.__decoration_colors.get('tongue'))
            # Draw eyes
            gfxdraw.filled_circle(self.__surf, int(x_coord_head + (self.__tile_width / 4)),  int(y_coord_head + (self.__tile_height / 2)), int(self.__tile_width / 8), (0, 0, 0))
            gfxdraw.filled_circle(self.__surf, int(x_coord_head + (self.__tile_width * 3 / 4)),  int(y_coord_head + (self.__tile_height / 2)), int(self.__tile_width / 8), (0, 0, 0))
        elif self.__direction == RIGHT:
            # Delete first quarter to make head smaller (and make space for tongue)
            gfxdraw.box(self.__surf, pygame.Rect(x_coord_head + self.__tile_width, y_coord_head, -(self.__tile_width / 4), self.__tile_height), self.__colors[EMPTY])
            # Draw tongue
            gfxdraw.box(self.__surf, pygame.Rect(x_coord_head + self.__tile_width, y_coord_head + (self.__tile_height * 3 / 8), -(self.__tile_width / 4), self.__tile_height / 4), self.__decoration_colors.get('tongue'))
            # Draw eyes
            gfxdraw.filled_circle(self.__surf, int(x_coord_head + (self.__tile_width / 2)),  int(y_coord_head + (self.__tile_height / 4)), int(self.__tile_width / 8), (0, 0, 0))
            gfxdraw.filled_circle(self.__surf, int(x_coord_head + (self.__tile_width / 2)),  int(y_coord_head + (self.__tile_height * 3 / 4)), int(self.__tile_width / 8), (0, 0, 0))
        elif self.__direction == DOWN:
            # Delete first quarter to make head smaller (and make space for tongue)
            gfxdraw.box(self.__surf, pygame.Rect(x_coord_head, y_coord_head, self.__tile_width, self.__tile_height / 4), self.__colors[EMPTY])
            # Draw tongue
            gfxdraw.box(self.__surf, pygame.Rect(x_coord_head + (self.__tile_width * 3 / 8), y_coord_head, self.__tile_width / 4, self.__tile_height / 4), self.__decoration_colors.get('tongue'))
            # Draw eyes
            gfxdraw.filled_circle(self.__surf, int(x_coord_head + (self.__tile_width / 4)),  int(y_coord_head + (self.__tile_height / 2)), int(self.__tile_width / 8), (0, 0, 0))
            gfxdraw.filled_circle(self.__surf, int(x_coord_head + (self.__tile_width * 3 / 4)),  int(y_coord_head + (self.__tile_height / 2)), int(self.__tile_width / 8), (0, 0, 0))
        elif self.__direction == LEFT:
            # Delete first quarter to make head smaller (and make space for tongue)
            gfxdraw.box(self.__surf, pygame.Rect(x_coord_head, y_coord_head, self.__tile_width / 4, self.__tile_height), self.__colors[EMPTY])
            # Draw tongue
            gfxdraw.box(self.__surf, pygame.Rect(x_coord_head, y_coord_head + (self.__tile_height * 3 / 8), self.__tile_width / 4, self.__tile_height / 4), self.__decoration_colors.get('tongue'))
            # Draw eyes
            gfxdraw.filled_circle(self.__surf, int(x_coord_head + (self.__tile_width / 2)),  int(y_coord_head + (self.__tile_height / 4)), int(self.__tile_width / 8), (0, 0, 0))
            gfxdraw.filled_circle(self.__surf, int(x_coord_head + (self.__tile_width / 2)),  int(y_coord_head + (self.__tile_height * 3 / 4)), int(self.__tile_width / 8), (0, 0, 0))


        # TODO draw body decorations
        temp_h, temp_w = self.__head_pos
        # Keep track of digestion
        d = len(self.__snake_path)
        # Add body
        for i, direction in enumerate(self.__snake_path):
            if direction == UP:
                temp_h -= 1
            elif direction == RIGHT:
                temp_w += 1
            elif direction == DOWN:
                temp_h += 1
            elif direction == LEFT:
                temp_w -= 1
            # Pacman effect
            temp_w = (temp_w + self.__board_width) % self.__board_width
            temp_h = (temp_h + self.__board_height) % self.__board_height

            # Coordinates of the corner
            x_coord = self.__tile_width * temp_w + self.__margin_left
            y_coord = self.__tile_height * (self.__board_height - 1 - temp_h) + self.__margin_top

            # print(self.__snake_path)
            if direction == DOWN:
                if i < len(self.__snake_path) - 1:
                    if self.__snake_path[i+1] == DOWN:
                        self.__draw_body_line(x_coord, y_coord, RIGHT)
                        self.__draw_body_line(x_coord, y_coord, LEFT)
                    elif self.__snake_path[i+1] == RIGHT:
                        self.__draw_body_line(x_coord, y_coord, LEFT)
                        self.__draw_body_line(x_coord, y_coord, DOWN)
                        self.__draw_body_point(x_coord, y_coord, [RIGHT, UP])
                    else: # LEFT
                        self.__draw_body_line(x_coord, y_coord, RIGHT)
                        self.__draw_body_line(x_coord, y_coord, DOWN)
                        self.__draw_body_point(x_coord, y_coord, [LEFT, UP])
                else: # If it's last tile
                    self.__draw_body_line(x_coord, y_coord, RIGHT)
                    self.__draw_body_line(x_coord, y_coord, LEFT)
                    self.__draw_body_line(x_coord, y_coord, DOWN)
            elif direction == RIGHT:
                if i < len(self.__snake_path) - 1:
                    if self.__snake_path[i+1] == RIGHT:
                        self.__draw_body_line(x_coord, y_coord, UP)
                        self.__draw_body_line(x_coord, y_coord, DOWN)
                    elif self.__snake_path[i+1] == UP:
                        self.__draw_body_line(x_coord, y_coord, RIGHT)
                        self.__draw_body_line(x_coord, y_coord, DOWN)
                        self.__draw_body_point(x_coord, y_coord, [LEFT, UP])
                    else: # DOWN
                        self.__draw_body_line(x_coord, y_coord, RIGHT)
                        self.__draw_body_line(x_coord, y_coord, UP)
                        self.__draw_body_point(x_coord, y_coord, [LEFT, DOWN])
                else: # If it's last tile
                    self.__draw_body_line(x_coord, y_coord, UP)
                    self.__draw_body_line(x_coord, y_coord, RIGHT)
                    self.__draw_body_line(x_coord, y_coord, DOWN)
            elif direction == UP:
                if i < len(self.__snake_path) - 1:
                    if self.__snake_path[i+1] == UP:
                        self.__draw_body_line(x_coord, y_coord, RIGHT)
                        self.__draw_body_line(x_coord, y_coord, LEFT)
                    elif self.__snake_path[i+1] == RIGHT:
                        self.__draw_body_line(x_coord, y_coord, LEFT)
                        self.__draw_body_line(x_coord, y_coord, UP)
                        self.__draw_body_point(x_coord, y_coord, [DOWN, RIGHT])
                    else: # LEFT
                        self.__draw_body_line(x_coord, y_coord, RIGHT)
                        self.__draw_body_line(x_coord, y_coord, UP)
                        self.__draw_body_point(x_coord, y_coord, [LEFT, DOWN])
                else: # If it's last tile
                    self.__draw_body_line(x_coord, y_coord, UP)
                    self.__draw_body_line(x_coord, y_coord, RIGHT)
                    self.__draw_body_line(x_coord, y_coord, LEFT)
            elif direction == LEFT:
                if i < len(self.__snake_path) - 1:
                    if self.__snake_path[i+1] == LEFT:
                        self.__draw_body_line(x_coord, y_coord, UP)
                        self.__draw_body_line(x_coord, y_coord, DOWN)
                    elif self.__snake_path[i+1] == UP:
                        self.__draw_body_line(x_coord, y_coord, LEFT)
                        self.__draw_body_line(x_coord, y_coord, DOWN)
                        self.__draw_body_point(x_coord, y_coord, [RIGHT, UP])
                    else: # DOWN
                        self.__draw_body_line(x_coord, y_coord, LEFT)
                        self.__draw_body_line(x_coord, y_coord, UP)
                        self.__draw_body_point(x_coord, y_coord, [RIGHT, DOWN])
                else: # If it's last tile
                    self.__draw_body_line(x_coord, y_coord, UP)
                    self.__draw_body_line(x_coord, y_coord, DOWN)
                    self.__draw_body_line(x_coord, y_coord, LEFT)

            # gfxdraw.box(self.__surf, pygame.Rect(x_coord, y_coord, self.__tile_width, self.__tile_height), (0, 0, 240))
            # gfxdraw.box(self.__surf, pygame.Rect(temp_w * self.__tile_width, temp_x * self.__tile_height, self.__tile_width/10, self.__tile_height/10), (0, 0, 255))



    def __draw_body_line(self, x, y, edge):
        if edge == RIGHT:
            gfxdraw.box(self.__surf, pygame.Rect(x + self.__tile_width, y, -self.__tile_width/5, self.__tile_height), self.__colors[HEAD_UP])
        elif edge == DOWN:
            gfxdraw.box(self.__surf, pygame.Rect(x, y, self.__tile_width, self.__tile_height / 5), self.__colors[HEAD_UP])
        elif edge == LEFT:
            gfxdraw.box(self.__surf, pygame.Rect(x, y, self.__tile_width / 5, self.__tile_height), self.__colors[HEAD_UP])
        elif edge == UP:
            gfxdraw.box(self.__surf, pygame.Rect(x, y + self.__tile_height, self.__tile_width, -self.__tile_height / 5), self.__colors[HEAD_UP])
        else:
            print('Invalid edge')


    def __draw_body_point(self, x, y, edges):
        if RIGHT in edges and UP in edges:
            # print('top_right_point')
            gfxdraw.box(self.__surf, pygame.Rect(x + self.__tile_width, y + self.__tile_height, -self.__tile_width / 5, -self.__tile_height / 5), self.__colors[HEAD_UP])
        elif RIGHT in edges and DOWN in edges:
            # print('bottom_right_point')
            gfxdraw.box(self.__surf, pygame.Rect(x + self.__tile_width, y, -self.__tile_width / 5, self.__tile_height / 5), self.__colors[HEAD_UP])
        elif LEFT in edges and UP in edges:
            # print('top_left_point')
            gfxdraw.box(self.__surf, pygame.Rect(x, y + self.__tile_height, self.__tile_width / 5, -self.__tile_height / 5), self.__colors[HEAD_UP])
        elif LEFT in edges and DOWN in edges:
            # print('bottom_left_point')
            gfxdraw.box(self.__surf, pygame.Rect(x, y, self.__tile_width / 5, self.__tile_height / 5), self.__colors[HEAD_UP])
        else:
            print('Invalid edge')


    def __draw_info_table(self):
        ## TODO change font size based on space available
        table_color = (181, 158, 94)
        text_color = (135, 7, 105)
        font_size = 24
        stats_margin_top = 10
        stats_margin_right = 10



        # Draw table background
        gfxdraw.box(self.__surf, pygame.Rect(self.__screen_width, self.__margin_top, self.__info_table_width * 0.85, self.__tile_height * len(self.__board)), table_color)

        # Set font
        font = pygame.font.Font(pygame.font.get_default_font(), font_size)

        # Create text
        texts = [
                f'Snake length: {len(self.__snake_path) + 1}',
                f'Targets eated: {len(self.__snake_path) + len(self.__digestion) - 1}',
                f'Total reward: {self.__total_reward}',
                f'Total steps: {self.__total_steps}'
                ]

        # Write on a new surface
        text_surfs = [font.render(text, True, text_color) for text in texts]

        # Flip (since will be flipped again after with the entire surface)
        text_surfs = [pygame.transform.flip(text_surf, False, True) for text_surf in text_surfs]

        # Place
        text_rects = [text_surf.get_rect() for text_surf in text_surfs]
        i = 1
        for text_rect in text_rects:
            text_rect.topleft = (self.__screen_width + stats_margin_right, self.__margin_top + self.__tile_height * len(self.__board) - stats_margin_top - font_size * i)
            i += 1


        # Draw
        for text_surf, text_rect in zip(text_surfs, text_rects):
            self.__surf.blit(text_surf, text_rect)


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
        # Decorations colors
        self.__decoration_colors = {'eyes': (0, 0, 0),
                                    'tongue': (255, 0, 0)}


    def close(self):
        if self.__screen is not None:
            self.__running = False
            # Wait in order for __check_closing to finish witout error
            # TODO replace timer with waiting for thread to exit
            time.sleep(0.5)
            pygame.display.quit()
            pygame.quit()


    # Play in human mode. Useful to debug
    def __play_human(self):
        self.__running = True
        done = False

        # Time to do a move
        self.__timeout = INITIAL_TIMEOUT

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
            if reward == REWARD_TARGET:
                self.__timeout *= TIMEOUT_DECREASE

            # Reset default move
            next_move = 0

        # Close env
        self.close()
        self.__screen = None
        self.__clock = None


    # Return a new env which is the exact same copy of the current one
    def clone(self):
        if self.__player == 'human':
            print('Can only clone environment for computers (for now at least)')
            return self
        # Generate new env
        # new_env = self.copy()
        new_env = SnakeEnv()
        # return new_env
        # Override values
        new_env.__board = self.__board.copy()
        new_env.__board_width = self.__board_width
        new_env.__board_height = self.__board_height
        new_env.__board_solid_border = self.__board_solid_border
        new_env.__board_type = self.__board_type
        new_env.__head_pos = self.__head_pos
        new_env.__snake_path = self.__snake_path.copy()
        new_env.__digestion = self.__digestion.copy()
        new_env.__direction = self.__direction
        # Override board copy
        new_env.__board_base = new_env.__board.copy()
        # Save reward mode
        new_env.__reward_mode = self.__reward_mode
        # Save state mode
        new_env.__state_mode = self.__state_mode
        # Save player mode
        new_env.__player = 'computer'

        # # Ensure actions are valid
        # new_env.__possible_actions = [0, 1, 2]
        # # Initialize variables
        new_env.__digestion = self.__digestion.copy()
        # # Keep track of steps
        new_env.__total_steps = self.__total_steps
        # # Keep track of the total reward
        new_env.__total_reward = self.__total_reward
        # # Screen for render
        # new_env.__screen = self.__screen.copy()
        # # Clock for render
        # new_env.__clock = self.__clock.copy()
        # # Screen width
        # new_env.__screen_width = 600
        # # Info table width
        # new_env.__info_table_width = 300
        # # Screen height
        # new_env.__screen_height = 700



        return new_env


    def deepcopy(self):
        return self.clone()
