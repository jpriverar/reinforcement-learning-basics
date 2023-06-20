import gym
import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler

def gather_samples(episodes):
    samples = []
    for episode in range(episodes):
        # Resetting the environment to start playing
        print(f"Starting episode {episode+1}")
        state, _ = env.reset()
        done = False

        # Starting to play
        while not done:
            # Getting a random action and playing it
            action = env.action_space.sample() # Encoded as integers, np.random.choice(2)
            next_state, reward, done, _, _= env.step(action)

            # Concatenating sample and saving the sample
            sample = np.append(state, action)
            samples.append(sample)

            # Shifting states in time
            state = next_state
    return samples

def epsilon_greedy_action(state, epsilon):
    # Getting random number between 0 a 1
    prob = np.random.random()

    # Exploration - random action
    if (prob < epsilon):
        action = env.action_space.sample()
    # Exploitation - greedy action
    else:
        action = np.argmax([model.predict(state, a) for a in np.arange(env.action_space.n)])
    return action

def learn_to_play(episodes, gamma, alpha, epsilon):
    # Save the rewards per episode
    rewards = []

    for episode in range(episodes):
        # To save the reward during the episode
        episode_reward = 0

        # Resetting the environment to start playing
        state, _ = env.reset()
        done = False

        # Starting to play
        while not done:
            # Getting an epsilon-greedy action and playing it
            action = epsilon_greedy_action(state, epsilon)
            next_state, reward, done, _, _= env.step(action)

            # Adding to the reward per episode
            episode_reward += reward

            # Action values for both states
            Qsa = model.predict(state, action)
            Qsa2 = max([model.predict(next_state, a) for a in np.arange(env.action_space.n)])

            # Define the target depending if it's a terminal state
            if done:
                y = reward
            else:
                y = reward + gamma*Qsa2

            # Updating the model weights
            model.w += alpha*(y - Qsa)*model.grad(state, action)

            # Shifting states in time
            state = next_state

        print(f"{episode+1} - {episode_reward}")
        rewards.append(episode_reward)

    return rewards

def play(episodes):
    rewards = []
    for episode in range(episodes):
        # To save the total reward during the episode
        episode_reward = 0

        # Resetting the environment to start playing
        state, _ = env.reset()
        done = False

        # Starting to play
        while not done:
            # Getting a greedy action and playing it
            action = epsilon_greedy_action(state, 0)
            next_state, reward, done, _, _= env.step(action)
            episode_reward += reward
            env.render()

            # Shifting states in time
            state = next_state

        rewards.append(episode_reward)
    return rewards

class Model:
    def __init__(self, samples):
        # Defining and training the feature expansion model
        self.featurizer = RBFSampler()
        self.featurizer.fit(samples)

        # Getting the dimentions of the featurizer to initialize weights
        dims = self.featurizer.random_offset_.shape[0]
        self.w = np.zeros(dims)

    def predict(self, state, action):
        input = np.append(state, action)
        x = self.featurizer.transform([input])[0]
        return np.matmul(self.w, x)

    def grad(self, state, action):
        input = np.append(state, action)
        x = self.featurizer.transform([input])[0]
        return x

if __name__ == "__main__":
    # Instantiating the Cart Pole Environment
    env = gym.make("CartPole-v1")

    # Playing randomly to collect samples and train the feature expansion kernel
    samples = gather_samples(10000)
    model = Model(samples)

    # Then play to learn
    learning_rate = 0.1
    discount_factor = 0.99
    epsilon = 0.5
    rewards = learn_to_play(1000, discount_factor, learning_rate, epsilon)

    # After training close the game
    env.close()

    # Plot the rewards per episode
    plt.plot(rewards)
    plt.grid()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per episode")
    plt.show()

    # Instantiating the Cart Pole Environment for actual playing
    env = gym.make("CartPole-v1", render_mode="human")
    rewards = play(1)
    env.close()

    # Plot the rewards during the exploitation
    plt.plot(rewards)
    plt.grid()
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per episode")
    plt.show()
