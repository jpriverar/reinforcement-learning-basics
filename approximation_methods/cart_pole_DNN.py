import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
import gym
    
class Agent:
    def __init__(self, env, samples):
        self.env = env
        self.gamma = 0.9
        
        self.scaler = StandardScaler()
        self.scaler.fit(samples)

        i = Input(env.observation_space.shape[0])
        x = Dense(5, activation="sigmoid")(i)
        outputs = []
        for _ in range(env.action_space.n):
            output = Dense(1)(x)
            outputs.append(output)

        self.model = Model(inputs=i, outputs=outputs)
        self.model.compile(optimizer='adam', loss='mse')

    def __transform_state(self, state):
        return self.scaler.transform([state])

    def predict(self, state):
        return self.model.predict(self.__transform_state(state), verbose=0)

    def act(self, state, epsilon):
        if np.random.random() < epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(state))
        
    def update(self, state, action, next_state, reward, done):
        # Defining the target of our update rule
        if done:
            y = reward
        else:
            y = reward + self.gamma*np.max(self.predict(next_state))

        target = np.array(self.predict(state))
        target[action][0] = y
        self.model.fit(state.reshape((1,4)), target.reshape((1,2)), verbose=0)

def play_one_episode(max_steps=5000, render=False, train=False, epsilon=0.1):
    episode_reward = 0
    steps = 0
    state, _ = env.reset()

    done = False
    while not done and steps < max_steps:
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps += 1

        if render: env.render()
        if train: 
            if done: reward = -300
            agent.update(state, action, next_state, reward, done)

        state = next_state
    return episode_reward

def gather_samples(episodes):
    samples = []
    for episode in range(episodes):
        state, _ = env.reset()
        done = False

        # Starting to play
        while not done:
            # Getting a random action and playing it
            action = env.action_space.sample() # Encoded as integers, np.random.choice(2)
            next_state, reward, done, _, _= env.step(action)
            samples.append(state)

            # Shifting states in time
            state = next_state
    return samples

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    samples = gather_samples(10)
    print(f"Collected {len(samples)} samples")

    agent = Agent(env, samples)
    episodes_to_train = 100
    rewards = np.zeros(episodes_to_train)

    epsilon = 0.9
    epsilon_min = 0.1
    epsilon_decay = 0.99

    for episode in range(episodes_to_train):
        print('Episode:', episode, 'Epsilon:', epsilon)
        episode_reward = play_one_episode(train=True, epsilon=epsilon)
        rewards[episode] = episode_reward
        epsilon = max(epsilon_min, epsilon*epsilon_decay)
    
    env.close()

    plt.plot(rewards)
    plt.title('Rewards per episode')
    plt.show()

    #env = gym.make('CartPole-v1', render_mode='human')
    #play_one_episode(render=True, epsilon=0)
    #env.close()