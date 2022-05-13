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

2. 
___

## Arguments








# TODO

- [ ] Add new shapes
  - [x] Shuriken
  - [x] Double
  - [x] Custom
  - [ ] Maze
  - [ ] Random
  - [ ] ...
- [ ] Improve initial placing of snake
- [ ] Improve graphic (draw snake border (!! RL agent can not see this! But probably not important))
- [ ] Create decent README
  - [ ] Describe requisite of custom board
  - [ ] Describe arguments (e.g. possible boards)
  - [ ] Show stats (e.g. points/length)
- [ ] Ev. add max moves to get target


# Done

- [x] Add pacman effect  
  Sometime already working :D, sometime NOT! (especially in checking new position)  
  I think it works if you first go left/up, because array index with negative number is possible (but not with number > max index)
- [x] Create basic env
- [x] Render in human mode
