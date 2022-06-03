README file for Snake gym environment!




[**Prerequisites**](#prerequisites) | [**Build**](#build) | [**Examples**](#execution-examples) | [**Arguments**](#arguments)

___

##  Prerequisites


* Clone this repository

* Python3.6 and pip for python 3

___

## Build

From main folder, register the new environment with:

```
$ pip install -e gym-snake
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

2. Generate an environment of size 10x10 without border (and pacman-effect). Play with keybord arrows (or WASD)
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


init / reset arguments:

name     default     possible values          requisites  
width=10, height=10, solid_border=True, shape='Normal', custom_board=None, player='computer', state_mode='states', reward_mode='normal', rewards=None

render arguments
- mode       'human'/'print'



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
            <td></td>
        </tr>
        <tr>
            <td>'human'</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=2>state_mode</td>
            <td rowspan=2>'states'</td>
            <td>'states'</td>
            <td></td>
        </tr>
        <tr>
            <td>'matrix'</td>
            <td></td>
        </tr>
        <tr>
            <td rowspan=3>reward_mode</td>
            <td rowspan=3>'normal'</td>
            <td>'normal'</td>
            <td></td>
        </tr>
        <tr>
            <td>'extended'</td>
            <td></td>
        </tr>
        <tr>
            <td>'adaptive'</td>
            <td></td>
        </tr>
        <tr>
            <td>rewards</td>
            <td>None</td>
            <td>None or dictionary</td>
            <td></td>
        </tr>
    </tbody>
</table>



# TODO

- [ ] Add images of the shapes
- [ ] Add new shapes
  - [x] Shuriken
  - [x] Double
  - [x] Custom
  - [ ] Maze
  - [ ] With random walls placed
  - [ ] ...
- [ ] Improve initial placing of snake
- [ ] Set an option to change the initial snake length
- [ ] Ev. add max moves to get target
