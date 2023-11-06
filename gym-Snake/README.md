README file for Snake gym environment!




[**Prerequisites**](#prerequisites) | [**Build**](#build) | [**Examples**](#execution-examples) | [**Arguments**](#arguments) | [**State representation**](#state-representation)

___

##  Prerequisites


* Clone this repository

* Python3.6 and pip for python 3

___

## Build

From main folder, register the new environment with:

```
$ pip install -e gym-Snake
```

Import the new environment in a python script using

```
import gym
import gym_Snake
```
___

## Execution examples

1. Generate environment with [**default parameters**](#arguments)
```
$ env = gym.make('Snake-v0')
```

2. Generate an environment of size 10x10 without border (and pacman-effect). Play with keyboard arrows (or WASD)
```
$ env = gym.make('Snake-v0', player='human', width=10, height=10, solid_border=False)
```

3. Generate an environment with the 'Shuriken' shape. Return the entire board (represented as a matrix) as observation
```
$ env = gym.make('Snake-v0', state_mode='matrix', shape='Shuriken')
```

4. Reset and existing environment maintaining all the parameters, but set the reward mode to 'extended'
```
$ env.reset(reward_mode="extended")
```
___

## Arguments

Arguments for the initialization/reset method

<table>
    <thead>
        <tr>
            <th>Argument name</th>
            <th>Default value</th>
            <th>Possible values</th>
            <th>Description</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>width</td>
            <td>10</td>
            <td>positive integer value</td>
            <td>The with of the board. Specific shapes have a minimal with</td>
        </tr>
        <tr>
            <td>height</td>
            <td>10</td>
            <td>positive integer value</td>
            <td>The height of the board. Specific shapes have a minimal height</td>
        </tr>
        <tr>
            <td>seed</td>
            <td>None</td>
            <td>integer between 0 and 2**32 - 1</td>
            <td>A seed for the random generator, in order to make experiments reproducibles</td>
        </tr>
        <tr>
            <td>solid_border</td>
            <td>True</td>
            <td>True or False</td>
            <td>Whether or not the border of the board are walls</td>
        </tr>
        <tr>
          <td rowspan=4>shape</td>
          <td rowspan=4>'Normal'</td>
          <td>'Normal'</td>
          <td>Classical rectangular shape, with possibly walls at the borders</td>
        </tr>
        <tr>
          <td>'Shuriken'</td>
          <td>A shape that resembles the Japanese "throwing stars"</td>
        </tr>
        <tr>
          <td>'Double_v'</td>
          <td>A normal board with a line of walls separating it horizontaly. Only a small pass is available to traverse from top half to bottom half</td>
        </tr>
        <tr>
          <td>'Double_v'</td>
          <td>Same as 'Double_h', but the separation is vertical</td>
        </tr>
        <tr>
            <td>custom_board</td>
            <td>None</td>
            <td>None or 2D matrix</td>
            <td>A matrix with the shape of the desired environment. The value 0 represents an empty tile, while a 1 represents a wall. If not None, the 'shape' parameter is overwritten</td>
        </tr>
        <tr>
            <td rowspan=2>player</td>
            <td rowspan=2>'computer'</td>
            <td>'computer'</td>
            <td>Normal mode. Play by calling the 'step' method and passing an action as argument</td>
        </tr>
        <tr>
            <td>'human'</td>
            <td>Human mode. Play with the keyboard while the game is rendered on a window</td>
        </tr>
        <tr>
            <td rowspan=2>state_mode</td>
            <td rowspan=2>'states'</td>
            <td>'states'</td>
            <td>The observation space consists in a value between 0 and 1023. For a more detailed explanation, see the "State representation" section</td>
        </tr>
        <tr>
            <td>'matrix'</td>
            <td>The observation returned by the step() and the reset() method is simply the entire board, as a 2D matrix</td>
        </tr>
        <tr>
            <td rowspan=3>reward_mode</td>
            <td rowspan=3>'normal'</td>
            <td>'normal'</td>
            <td>The rewards are given only when the snake eats a target, or when it dies</td>
        </tr>
        <tr>
            <td>'extended'</td>
            <td>In addition to the normal rewards, a positive reward is given to the snake for each step reducing its (l1) distance from the target, and a negative reward for each step increasing it. It is also possible to give an additional reward for each step in which the snake does not die</td>
        </tr>
        <tr>
            <td>'adaptive'</td>
            <td>Same as 'extended', but the rewards for approaching/walking away from the target are decreased the longer the snake is. The intuition behind this is that when the snake is longer, going straight to the target becomes less and less important, and rather more focus should be put in surviving</td>
        </tr>
        <tr>
            <td>rewards</td>
            <td>None</td>
            <td>None or dictionary</td>
            <td>It is possible to override the default value of the rewards by passing a dictionary. The keys in this dictionary are all optional, and are: "REWARD_TARGET", "REWARD_COLLISION", "REWARD_TOWARD", "REWARD_AWAY", and "REWARD_SURVIVED"</td>
        </tr>
    </tbody>
</table>


## State representation

To use the environment with classical RL algorithms (e.g. Q-Learning or SARSA), instead of the entire matrix it is possible to receive only a value between 0 and 1023 as observation. This value encodes the local information around the snake head, and is the decimal representation of a binary array defined as following:

0-2 bits: target position relative to the head (8 possibilities)  
3-9 bits: obstacle in front or not, obstacle in front-right or not, obstacle on the right or not, ... (obstacle behind always true, so ignore)

The step() method returns also a dictionary of information, often called 'info'. It is possible to visualize the position of the target and the wall obstacle around the head as 2D matrices from this dictionary, using

```python
info["target_pos"]
info["walls"]
```


# TODO - Future works

- [ ] Add images of the shapes
- [ ] Add new shapes
  - [ ] Maze
  - [ ] With random walls placed
  - [ ] ...
- [ ] Improve initial placing of snake
- [ ] Set an option to change the initial snake length
- [ ] Ev. add max moves to get target
