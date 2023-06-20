import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
import gym

class Agent:
    def __init__(self, env):
        self.env = env
        self.gamma = 0.9

        # Collecting state samples
        n_samples = 100
        samples = np.zeros((n_samples, env.observation_space.shape[0]))
        for i in range(n_samples):
            samples[i] = env.observation_space.sample()

        # Training a scaler object to scale all future states
        self.scaler = StandardScaler()
        scaled_samples = self.scaler.fit_transform(samples)

        # To expand states into features
        self.featurizer = RBFSampler()
        self.featurizer.fit(scaled_samples)

        # Linear model for each possible action
        self.Q = [SGDRegressor() for _ in range(env.action_space.n)]

        # Initializing model weights with dummy value
        state = self.__transform_state(env.reset()[0])

        for q in self.Q:
            q.partial_fit(state, [0]) # 0 as target in this case allows for optimistic values

    def __transform_state(self, state):
        scaled_state = self.scaler.transform([state])
        return self.featurizer.transform(scaled_state)
    
    def predict(self, state):
        state = self.__transform_state(state)
        return np.array([q.predict(state) for q in self.Q])

    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state))

    def update(self, state, action, next_state, reward, done):
        # Target definition
        if done:
            y = reward
        else:
            y = reward + self.gamma*np.max(self.predict(next_state))

        # One step of gradient descent
        state = self.__transform_state(state)
        self.Q[action].partial_fit(state, [y])

def play_one_episode(max_steps=1000, render=False, train=False, epsilon=0.1):
    state, _ = env.reset()
    episode_reward = 0
    steps = 0

    done = False
    while not done and steps < max_steps:
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps += 1

        if render: env.render()
        if train: agent.update(state, action, next_state, reward, done)
        state = next_state

    return episode_reward

if __name__ == "__main__":

    env = gym.make("MountainCar-v0")
    agent = Agent(env)

    episodes_to_train = 200
    rewards = np.zeros(episodes_to_train)

    epsilon = 0.9
    epsilon_min = 0.1
    epsilon_decay = 0.999

    for i in range(episodes_to_train):
        print("Episode:", i, "Epsilon:", epsilon)
        episode_reward = play_one_episode(train=True, epsilon=epsilon)
        rewards[i] = episode_reward
        epsilon = max(epsilon_min, epsilon*epsilon_decay)

    plt.plot(rewards)
    plt.title('Reward per episode')
    plt.show()
    env.close()

    env = gym.make("MountainCar-v0", render_mode='human')
    play_one_episode(render=True, train=False, epsilon=0)
    env.close()