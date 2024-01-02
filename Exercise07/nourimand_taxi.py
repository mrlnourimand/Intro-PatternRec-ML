"""
DATA.ML.100: Introduction to Pattern Recognition and Machine Learning
Ex 07, title:Reinforcement learning (Gymnasium).

I used Gym environment in this program. There is a map for a taxi movement.
here is the description from the website:
https://www.gymlibrary.dev/environments/toy_text/taxi/
There are 6 discrete deterministic actions:(south, north, east, west, pickup and
drop passenger). Our task is to implement Q-learning to solve the Taxi problem
with optimal policy.

Creator: Maral Nourimand
Student id number: 151749113
Email: maral.nourimand@tuni.fi
"""

# Load OpenAI Gym and other necessary packages
import gym
import random
import numpy as np
import time


def main():
    # Environment
    env = gym.make("Taxi-v3", render_mode='ansi')

    # Training parameters for Q learning
    alpha = 0.9  # Learning rate
    gamma = 0.9  # Future reward discount factor
    num_of_episodes = 1000
    num_of_steps = 500  # per each episode

    # Q tables for rewards
    # Q_reward = -100000*numpy.ones((500,6)) # All same
    Q_reward = -1 * np.random.random((500, 6))  # Random

    # Training w/ random sampling of actions
    for episode in range(num_of_episodes):
        state = env.reset()[0]
        tot_reward = 0
        for t in range(num_of_steps):
            action = np.argmax(Q_reward[state, :])

            next_state, reward, done, truncated, info = env.step(action)

            # Q-learning update rule
            Q_reward[state, action] = (1 - alpha) * Q_reward[
                state, action] + alpha * (reward + gamma * np.max(
                Q_reward[next_state, :]))
            # Q_reward[state, action] = (1) * Q_reward[
            #     state, action] + alpha * (reward + gamma * np.max(
            #     Q_reward[next_state, :]))

            state = next_state
            tot_reward += reward
            # print(env.render())

            if done:
                break

    # Testing
    total_rewards = []
    total_actions = []
    for _ in range(10):
        state = env.reset()[0]
        tot_reward = 0
        num_actions = 0
        while True:
            action = np.argmax(Q_reward[state, :])
            state, reward, done, truncated, info = env.step(action)
            tot_reward += reward
            num_actions += 1
            print(env.render())
            time.sleep(1)
            if done:
                break
        total_rewards.append(tot_reward)
        total_actions.append(num_actions)

    # Compute and print average total reward and average number of actions
    avg_reward = sum(total_rewards) / len(total_rewards)
    avg_actions = sum(total_actions) / len(total_actions)
    print("Average Total Reward:", avg_reward)
    print("Average Number of Actions:", avg_actions)


if __name__ == "__main__":
    main()
