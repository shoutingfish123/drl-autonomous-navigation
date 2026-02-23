#! /usr/bin/env python

import numpy as np
import tensorflow as tf
from env_turtlebot import GameState
import random
from ddpg_brain import (
    buffer, 
    policy, 
    ou_noise, 
    actor_model, 
    critic_model, 
    target_actor, 
    target_critic, 
    update_target, 
    tau
)
import matplotlib.pyplot as plt

# 1. Initialize the Environment (The Body)
# env is an instance of the gamestate class
env = GameState()

# 2. Parameters
total_episodes = 1000    # How many games to play
max_steps = 500         # Max steps per game (prevents getting stuck)

# To store reward history of each episode
ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []

print("--- Starting Training ---")


for ep in range(total_episodes):

    # A. Reset the environment for a new game
    prev_state = env.reset()
    ou_noise.reset()        # to reset the noise at each episode, to avoid continuous safe motion like driving in circles
    episodic_reward = 0

    print("Target x coordinate: ",env.target_x)
    print("Target y coordinate: ",env.target_y)
    print("episode number: ",ep)
    # changes each episode
    
    for step in range(max_steps):
        # B. Get Action from the Brain
        # We pass the state and the noise generator
        tf_prev_state = tf.reshape(tf.convert_to_tensor(prev_state), [1,28])
        # lets promote erratic motion in the first 10-20 episodes
        if ep<20:
            # Create random action: 
            # Action 0 (Linear): Random between -0.5 and 0.5
            # Action 1 (Angular): Random between -1.0 and 1.0
            action = [random.uniform(-0.5, 0.5), random.uniform(-1.0, 1.0)]
        else:
            action = policy(tf_prev_state, ou_noise)
        
        # Debug print to see what the brain is thinking (Optional)
        # print(f"Action: Lin={action[0]:.2f}, Ang={action[1]:.2f}")

        # C. Act on the Environment (The Body)
        # Recieve state and reward from environment.
        # action[0] is linear, action[1] is angular
        reward, state, done = env.game_step(0.1, action[0], action[1])

        # D. Save to Buffer (The Memory)
        # Note: We use [0] because the Env returns shape (1,28) but Buffer wants (28,)
        buffer.record((prev_state[0], action, reward, state[0]))
        episodic_reward += reward

        # E. Train the Brain (The Learning)
        buffer.learn()
        
        # F. Update Target Networks (The Stability)
        update_target(target_actor, actor_model, tau)
        update_target(target_critic, critic_model, tau)

        # G. Update state for next step
        prev_state = state

        # Check if game is over (Crash or Goal)
        if done:
            break

    ep_reward_list.append(episodic_reward)

    # Mean of last 40 episodes
    avg_reward = np.mean(ep_reward_list[-40:])
    print(f"Episode * {ep} * Avg Reward is ==> {avg_reward:.2f}")
    avg_reward_list.append(avg_reward)

    # Save the weights every 10 episodes so you don't lose progress
    if ep % 10 == 0:
        actor_model.save_weights("turtlebot_actor.weights.h5")
        critic_model.save_weights("turtlebot_critic.weights.h5")

# Save final weights
actor_model.save_weights("turtlebot_actor_final.weights.h5")
critic_model.save_weights("turtlebot_critic_final.weights.h5")
print("--- Training Complete ---")

# Plotting the learning curve (reward vs no. of episodes)
plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Reward")
plt.title("DDPG Training Performance")
plt.savefig("training_result.png")
plt.show()