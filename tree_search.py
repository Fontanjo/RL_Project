import gym
import gym_Snake
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from multiprocessing import Pool
import multiprocessing.pool

# Normal Pool can not be nested
class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess

# We sub-class multiprocessing.pool.Pool instead of multiprocessing.Pool
# because the latter is only a wrapper function, not a proper class.
class NestablePool(multiprocessing.pool.Pool):
    def __init__(self, *args, **kwargs):
        kwargs['context'] = NoDaemonContext()
        super(NestablePool, self).__init__(*args, **kwargs)



def pool_tree_search_best_action(env, dept, target_reward, pooling_condition=np.inf):
    actions = [0, 1, 2]
    envs = [env.clone() for _ in actions]

    # Play each action
    obss, rewards, dones, _ = np.array([e.step(a) for e,a in zip(envs, actions)]).T


    with NestablePool(3) as p:
        p.daemon = False
        res = p.starmap(pool_tree_search_best_action_rec, [[envs[i], dept - 1, [i], target_reward, pooling_condition] for i in actions])
        results, act = np.array(res).T


        # vals = [0.9 * rs + rw for rs, rw in zip(results, rewards)]
        # choices = [i for i, r in enumerate(vals) if r == max(vals)]

        # c = np.random.choice(choices)

        # print(results)
        # print(choices)
        # print(c, np.argmax(vals))

        # return c
        return np.argmax([0.9 * rs + rw for rs, rw in zip(results, rewards)])



stop = False
def pool_tree_search_best_action_rec(env, dept, acts, target_reward, pooling_condition=np.inf, dept_after_reward=4):
    global stop
    actions = [0, 1, 2]

    # Clone envs
    envs = [env.clone() for _ in actions]

    # Play each action
    obss, rewards, dones, _ = np.array([e.step(a) for e,a in zip(envs, actions)]).T

    # Termination case
    if dept == 0:
        valids = [i for i, x in enumerate(rewards) if x == max(rewards)]
        i = np.random.choice(valids)
        return rewards[i], i

    remaining_dept = [dept - 1 for _ in actions]

    # Mark as done loops that have already reached the target
    # Allows to slightly accellerate the algorithm, but suffers from the same
    # problems of Q-learning and SARSA (snake traps itself)
#     for i, r in enumerate(rewards):
#         if r >= target_reward:
#             dones = [True for _ in range(len(dones))] # Stop search for this branch
#             stop = True # Send the stop signal to all branches
#             dones[i] = True # Stop search for given path
#             remaining_dept = [0 for _ in actions] # Stop search for other paths, limit for given one
#             remaining_dept[i] = min(dept_after_reward, remaining_dept[i]) # Limit the search for the given path

    # Check if external stop - makes it very fast limiting dept in most cases
    if stop:
        dones = [True for _ in range(len(dones))]


    # Multiprocess if still high in tree
#     if pooling_condition(dept):
    if dept > pooling_condition:
        with NestablePool(3) as p:
            p.daemon = False
            res = p.starmap(pool_tree_search_best_action_rec, [[envs[i], dept - 1, acts + [i], target_reward, pooling_condition] for i in actions])
            results, act = np.array(res).T
    else:
        results, _ = np.array([tree_search_best_action_rec(e, dept-1, acts + [i], target_reward=target_reward) if not dones[i] else (0, -1) for i, e, d in zip(range(len(actions)), envs, remaining_dept)]).T

#     results, _ = np.array([tree_search_best_action_rec(e, dept-1, acts + [i], target_reward=target_reward) if not dones[i] else (0, -1) for i, e in enumerate(envs)]).T
#     results = [tree_search_best_action_rec(e, dept-1, acts + [i]) for i, e in enumerate(envs)]


    # Add reward MULTIPLIED by dept. In this way, sort of discount rewards more far away
    # Without this trick, it is possible that the snake turn around the target without ever eating it, since a
    # position in which we can eat the target in N steps has the same value as the one in which we eat it
    results = [results[i] + rewards[i] * (1 + dept/1000) for i in range(len(results))]
#     results = [results[i] + rewards[i] for i in range(len(results))]


    # If multiple results with same (max) value, choose random among them
    valids = [i for i, x in enumerate(results) if x == max(results)]

    # Randomly choose one of the best actions
    i = np.random.choice(valids)

    return results[i], i


stop = False
def tree_search_best_action_rec(env, dept, acts, target_reward, dept_after_reward=4):
    global stop
    actions = [0, 1, 2]

    # Clone envs
    envs = [env.clone() for _ in actions]

    # Play each action
    obss, rewards, dones, _ = np.array([e.step(a) for e,a in zip(envs, actions)]).T

    # Termination case
    if dept == 0:
        valids = [i for i, x in enumerate(rewards) if x == max(rewards)]
        i = np.random.choice(valids)
        return rewards[i], i

    remaining_dept = [dept - 1 for _ in actions]

    # Mark as done loops that have already reached the target
    # Allows to slightly accellerate the algorithm, but suffers from the same
    # problems of Q-learning and SARSA (snake traps itself)
#     for i, r in enumerate(rewards):
#         if r >= target_reward:
#             dones = [True for _ in range(len(dones))] # Stop search for this branch
#             stop = True # Send the stop signal to all branches
#             dones[i] = True # Stop search for given path
#             remaining_dept = [0 for _ in actions] # Stop search for other paths, limit for given one
#             remaining_dept[i] = min(dept_after_reward, remaining_dept[i]) # Limit the search for the given path

    # Check if external stop - makes it very fast limiting dept in most cases
    if stop:
        dones = [True for _ in range(len(dones))]


    results, _ = np.array([tree_search_best_action_rec(e, dept-1, acts + [i], target_reward=target_reward) if not dones[i] else (0, -1) for i, e, d in zip(range(len(actions)), envs, remaining_dept)]).T
#     results, _ = np.array([tree_search_best_action_rec(e, dept-1, acts + [i], target_reward=target_reward) if not dones[i] else (0, -1) for i, e in enumerate(envs)]).T
#     results = [tree_search_best_action_rec(e, dept-1, acts + [i]) for i, e in enumerate(envs)]


    # Add reward MULTIPLIED by dept. In this way, sort of discount rewards more far away
    # Without this trick, it is possible that the snake turn around the target without ever eating it, since a
    # position in which we can eat the target in N steps has the same value as the one in which we eat it
    results = [results[i] + rewards[i] * (1 + dept/1000) for i in range(len(results))]
#     results = [results[i] + rewards[i] for i in range(len(results))]


    # If multiple results with same (max) value, choose random among them
    valids = [i for i, x in enumerate(results) if x == max(results)]

    # Randomly choose one of the best actions
    i = np.random.choice(valids)

    return results[i], i



def play_epoch(env, render = False, dept=5, target_reward = 10, sleep_time = 0.5, max_step = 100, pooling_condition = np.inf):

    # Reset env
    obs = env.reset()

    done = False

    # Sum the rewards
    total_rew = 0

    i = 0
    while not done:
        # Show
        if render: env.render()
        # Choose next action
#         new_act = tree_search_best_action(env, dept=dept, target_reward=target_reward)
        new_act = pool_tree_search_best_action(env, dept=dept, target_reward=target_reward, pooling_condition=pooling_condition)
        # print(new_act)
        # Act in the env
        obs, reward, done, info = env.step(new_act)
        # Store reward
        total_rew += reward
        # Slow render
        if render and sleep_time > 0: time.sleep(sleep_time)
        i += 1
        if i == max_step: break

    # Return total reward
    return total_rew


def main():
    custom_rewards = {
        "REWARD_TARGET": 10,
        "REWARD_COLLISION": -100,
        "REWARD_TOWARD": 1,
        "REWARD_AWAY": -1
    }

    env = gym.make('Snake-v0',
                   player = 'computer',
                   shape = 'Shuriken',
                   state_mode = 'states',
                   reward_mode = 'normal',
                   width = 10,
                   height = 10,
                   solid_border = True,
                   rewards = custom_rewards)

    i = 7
    cond = 5
    start = time.time()
    r = play_epoch(env = env, render = True, dept=i, target_reward=10, sleep_time = 0, max_step = 1000, pooling_condition = cond)
    stop = time.time()
    print(f'i = {i}, total reward = {r}, time = {round(stop-start, 2)}')


if __name__ == "__main__":
    main()
