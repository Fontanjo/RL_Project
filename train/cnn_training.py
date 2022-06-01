import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import random
import time
from tqdm import tqdm
import argparse
import gym_Snake

from keras import layers,Model,optimizers,losses
import tensorflow as tf

model = None
model_target = None
replay_buffer = None

def create_small_Q_network(input_matrix,number_of_action):

    inputs = layers.Input(shape=input_matrix)

    # CNN layers
    layer1 = layers.Conv2D(32, kernel_size=(3, 3),activation="relu")(inputs)
    layer2 = layers.Conv2D(64, kernel_size=(3, 3),activation="relu")(layer1)
    #layer3 = layers.Conv2D(128, kernel_size=(3, 3),activation="relu")(layer2)

    # Max Pooling which can probably be removed 
    #layer4 = layers.MaxPooling2D(pool_size=(2,2))(layer3)

    # Flatten 
    layer5 = layers.Flatten()(layer2)

    # Dense layer to predict the action
    layer6 = layers.Dense(128,activation="relu")(layer5)
    action = layers.Dense(number_of_action,activation="softmax")(layer6)

    return Model(inputs=inputs,outputs=action)


def play_epoch(env,frame_number,epoch_number, loss_function,optimizer,
discount = 0.99,max_steps_per_epoch = 200,epsilon_min = 0.0002,
explo_games = 5000,epsilon_games = 15000,render = False):

    global model
    global model_target
    global replay_buffer

    number_of_action = env.action_space.n
    state = env.reset().repeat(2,axis=0).repeat(2,axis=1)
    
    episode_reward = 0
    
    done = False

    step_counter = 0

    #################################
    ## Change to modify the model training
    
    # How many frame to recover from the replay buffer
    batch_size = 64
    # Train the model after 4 actions
    update_Q_network = 4
    # How often to update the target network
    update_target_network = 10000
    # Replay buffer size
    max_memory_length = 400000

    # Decay the epsilon only after each new games, rather than each frames of the games, probably not too relevant

    ###############################
    # Epsilon decay

    # Decay probability of taking random action
    epsilon = 1. - (epoch_number / epsilon_games)
    epsilon = max(epsilon, epsilon_min)

    while not done and step_counter < max_steps_per_epoch:
        
        step_counter += 1
        frame_number += 1

        ###############################
        # Show
        
        if render: env.render()

        ###############################
        # Select Action

        # Use epsilon greedy for exploration, 
        # also in the beginning for "epsilon_random_frames" always random
        if epoch_number < explo_games or epsilon > random.random():
            # Take random action
            action = np.random.choice(number_of_action)
        else:
            # Predict action Q-values
            # From environment state
            state_tensor = tf.convert_to_tensor(state)
            state_tensor = tf.expand_dims(state_tensor, 0)
            action_probs = model(state_tensor, training=False)
            # Take best action
            action = tf.argmax(action_probs[0]).numpy()

        

        ###############################
        # Act in the env

        state_next, reward, done, _ = env.step(action)
        state_next = state_next.repeat(2,axis=0).repeat(2,axis=1)

        ###############################
        # Store reward for that run

        episode_reward += reward    

        ###############################
        # Slow render
        if render: time.sleep(0.5)

        ###############################
        # Save actions and states in replay buffer
        
        replay_buffer.append([state,action,reward,state_next,done])
 
        state = state_next

        ###############################
        # Update the Q-Network

        # Only once every 4 frames (good tradeoff between speed and efficiency of training)
        # and only when the buffer is at least 32 (the size of the batch)
        
        # Update every fourth frame and once batch size is over 32
        if frame_number % update_Q_network == 0 and len(replay_buffer) > batch_size:

            
            # Get indices of samples for replay buffers
            samples = random.sample(replay_buffer,batch_size)

            # Sample from replay buffer
            action_sample = []
            state_sample = []
            state_next_sample = []
            rewards_sample = []
            done_sample = []

            for sample in samples:
                s, a, r, s_t, d = sample
                action_sample.append(a)
                state_sample.append(s)
                state_next_sample.append(s_t)
                rewards_sample.append(r)
                done_sample.append(float(d))

            state_sample = np.array(state_sample)
            state_next_sample = np.array(state_next_sample)
            done_sample = tf.convert_to_tensor(done_sample)

            # Build the updated Q-values for the sampled future states
            # Use the target model for stability
            future_rewards = model_target.predict(state_next_sample)
            # Q value = reward + discount factor * expected future reward
            updated_q_values = rewards_sample + discount * tf.reduce_max(future_rewards, axis=1)

            # If final frame set the last value to -1
            updated_q_values = updated_q_values * (1 - done_sample) - done_sample

            # Create a mask so we only calculate loss on the updated Q-values
            masks = tf.one_hot(action_sample, number_of_action)

            with tf.GradientTape() as tape:
                # Train the model on the states and updated Q-values
                q_values = model(state_sample)

                # Apply the masks to the Q-values to get the Q-value for action taken
                q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
                # Calculate loss between new Q-value and old Q-value
                loss = loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        ###############################
        # Update the target-Network

        if frame_number % update_target_network == 0:
            
            
            # update the the target network with new weights
            model_target.set_weights(model.get_weights())

        ###############################
        # Delete the 1'000 oldest replays from buffer history when bigger than allowed 

        if len(replay_buffer) > max_memory_length:
            
            del replay_buffer[:1000]
        

    return episode_reward,frame_number


def main(args):

    global model
    global model_target
    global replay_buffer

    nbr_epoch = args.epoch 
    shape = args.shape

    loss_function = losses.Huber()
    optimizer = optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)

    env_reward = {"REWARD_TARGET":1,
        "REWARD_COLLISION":-1,
        "REWARD_TOWARD":1/5,
        "REWARD_AWAY":-1/5}

    env = gym.make('Snake-v0', 
               player='computer', 
               shape=shape, 
               state_mode='matrix', 
               reward_mode = 'adaptive', 
               width=10, 
               height=10, 
               solid_border=True,
               rewards=env_reward)

    number_of_action = env.action_space.n
    x = 2*int(np.sqrt(env.observation_space.n))
    input_matrix = (x,x,1,)

    model = create_small_Q_network(input_matrix=input_matrix,number_of_action=number_of_action)

    model_target = create_small_Q_network(input_matrix=input_matrix,number_of_action=number_of_action)

    # Experience replay buffers
    replay_buffer = []


    episode_reward_history = []
    
    # Using huber loss for stability

    # number of epoch
    #nb_iterations = 50000
    #nb_iterations = 30000
    #nb_iterations = 10

    frame_number = 0


    print("start training")
    for i in tqdm(range(nbr_epoch)):

        r_episode,frame_number = play_epoch(env=env,frame_number=frame_number,
        epoch_number=i,loss_function=loss_function,optimizer=optimizer)
        episode_reward_history.append(r_episode)
        

        if i % 1000 == 0:

            print("Epoch {}: Reward over the last 50 rounds {}".format(i,np.array(episode_reward_history[-50:]).mean()))
            model.save("./networks/10x_{}_{}_smallQ".format(shape,nbr_epoch))
            df = pd.DataFrame(episode_reward_history)
            df.to_csv("./training_reward/10x_{}_{}_smallQ.csv".format(shape,nbr_epoch))
    
    model.save("./networks/final_10x_{}_{}_smallQ".format(shape,nbr_epoch))

    df = pd.DataFrame(episode_reward_history)
    df.to_csv("./training_reward/final_10x_{}_{}_smallQ.csv".format(shape,nbr_epoch))


if __name__ == "__main__":
    # Ev. extract parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--shape", type=str, default="Classic", help="The arena shape: Classic, Shuriken, Double_h or Double_v")
    parser.add_argument("--epoch", type=int, default=50000, help="The number of epoch")
    args = parser.parse_args()
    main(args)
