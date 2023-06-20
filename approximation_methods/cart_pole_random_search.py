import numpy as np
import matplotlib.pyplot as plt
import gym

def play_one_episode(max_steps=10000, render=False):
    episode_reward = 0
    steps = 0
    state, _ = env.reset()

    done = False
    while not done and steps < max_steps:
        action = get_action(state)
        state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps += 1

        if render: env.render()
    
    return episode_reward

def test_weights_performance(test_episodes):
    rewards = np.zeros(test_episodes)
    for episode in range(test_episodes):
        episode_reward = play_one_episode()
        rewards[episode] = episode_reward
    return np.mean(rewards)

def get_action(state):
    return 1 if state.dot(w) > 0 else 0

if __name__ == '__main__':
    env = gym.make('CartPole-v1')   

    best_reward = 0
    weights_to_try = 100
    episodes_per_weights = 10

    avg_rewards = np.zeros(weights_to_try)
    for i in range(weights_to_try):
        print(f"Trying weights number {i}")
        w = np.random.rand(env.observation_space.shape[0])
        avg_reward = test_weights_performance(episodes_per_weights)
        avg_rewards[i] = avg_reward

        # Check it this weights yield better performance
        if avg_reward > best_reward:
            best_w = w
            best_reward = avg_reward

    plt.plot(avg_rewards)
    plt.title('Average reward per weights')
    plt.show()

    env.close()

    env = gym.make('CartPole-v1', render_mode='human') 

    # Trying again some episodes with the best weights
    w = best_w
    episodes_to_test = 1
    rewards = np.zeros(episodes_to_test)

    for episode in range(episodes_to_test):
        episode_reward = play_one_episode(render=True)
        rewards[episode] = episode_reward

    plt.plot(rewards)
    plt.title('Reward per episode (Best weights)')
    plt.show()
