import gym
from gym import error, spaces, utils
from gym.utils import seeding
import numpy as np
import math

EMPTY = 0
WALL = 1
BODY = 2
HEAD = 3
TARGET = 4
UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3
COLLISION_REWARD = -10
TARGET_REWARD = 1
SURVIVED_REWARD = 0

class SnakeEnv(gym.Env):
    metadata = {'render.modes': ['human']}


    def __init__(self, width=10, height=10, solid_border=True):
        # Store informations
        self.__board_width = width
        self.__board_height = height
        self.__board_solid_border = solid_border
        # Initialize variables
        self.reset()


    # Actions: 0 continue, 1 turn right, 2 turn left
    def step(self, action):
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
            self.__digestion.append(len(self.__snake_path))
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


    def reset(self, width=None, height=None, solid_border=None):
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


    # Generate the board on which the snake moves
    def __generate_board(self):
        # Generate empty matrix
        b = np.zeros((self.__board_height, self.__board_width))
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
        self.__snake_path = [RIGHT, RIGHT, UP, RIGHT, DOWN, DOWN, DOWN, LEFT]


    # Remove the snake from the board (before re-placing it)
    def __remove_snake(self):
        # Remove head
        self.__board[self.__head_pos] = EMPTY
        # Remove body
        self.__board[self.__board == BODY] = EMPTY


    # Place the snake on the board, based on the snake_path
    def __place_snake(self):
        # Place the head
        self.__board[self.__head_pos[0], self.__head_pos[1]] = HEAD
        # Store coordinates
        temp_h, temp_w = self.__head_pos
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
            self.__board[temp_h, temp_w] = BODY


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
        # Insert a body tile behing the head
        self.__snake_path.insert(0, (direction + 2) % 4)
        # If necessary, remove last part
        if 0 not in self.__digestion:
            self.__snake_path.pop()
        # Else, finish digestion
        else:
            self.digestion.remove(0)


    # Place the target
    # Randomly choose one of the empty slots
    def __place_target(self):
        # Get empty tiles
        possible = np.where(self.__board == EMPTY)
        # If any empty tile found, place a target
        if len(possible) >= 0:
            pos = np.random.randint(len(possible[0]))
            self.__board[possible[0][pos], possible[1][pos]] = TARGET


    def render(self, mode='human', close=False):
        print(self.__board)
