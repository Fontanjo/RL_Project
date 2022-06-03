# RL_Project


### Carrel Vincent, Fontana Jonas


---


This page contains our code for the Reinforcement Learning and Decision Making under Uncertainty project, under the supervision of Pr Christos Dimitrakakis

---

# Subject

In this project, we build a new Gym environment representing the famous Nokia game Snake. We implement 4 different algorithms (SARSA, Q-Learning, TreeSearch and Deep Q-Learning) and compare the performances of the different models in different arena. Then we pay a closer look at the capabilities of the Deep Q-Learning methods to extrapolate to new arenas. We discuss the shortcomings in the training setup and propose different avenue to consolidate the model and obtain an agent able to perform in different environments.

---

# TODO

- [ ] Train until convergence/way more steps than 50k
- [ ] Implement two/multiple different replay buffer (based on the reward obtained, the length of the snake, etc...) to ensure the presence of "important but rare" situation in the sampling
- [ ] Train using a random initial length of the snake
- [ ] Train with different (random) walls position

---
 
# Related Work

[https://www.youtube.com/watch?v=i0Pkgtbh1xw](https://www.youtube.com/watch?v=i0Pkgtbh1xw)  
YouTube video, works on pixels using DeepQL

[https://github.com/DragonWarrior15/snake-rl](https://github.com/DragonWarrior15/snake-rl)  
Git repo with similar work

[https://sid-sr.github.io/Q-Snake/](https://sid-sr.github.io/Q-Snake/)  
Visualization of Q-learning

[https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36](https://towardsdatascience.com/snake-played-by-a-deep-reinforcement-learning-agent-53f2c4331d36)  
Interesting analysis of various part. Still works on "close neighborhood". Would be interesting to expand and work on the whole board (Deep Q-Learning)

[https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/](https://www.geeksforgeeks.org/ai-driven-snake-game-using-deep-q-learning/)  
Deep QL, same action encoding as our

[https://www3.hs-albsig.de/wordpress/point2pointmotion/2020/10/09/deep-reinforcement-learning-with-the-snake-game/](https://www3.hs-albsig.de/wordpress/point2pointmotion/2020/10/09/deep-reinforcement-learning-with-the-snake-game/)  
Based on the code that solved the atari games. However do not obtain incredible results
