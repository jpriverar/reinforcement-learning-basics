import numpy as np
import matplotlib.pyplot as plt
import gym

class StateTransformer:
    def __init__(self):
        car_pos_bins = np.linspace(-0.21, 0.21, 9)
        car_vel_bins = np.linspace(-1.5, 1.5, 9)
        pole_pos_bins = np.linspace(-0.23, 0.23, 9)
        pole_vel_bins = np.linspace(-2, 2, 9)
        self.state_feature_bins = [car_pos_bins, car_vel_bins, pole_pos_bins, pole_vel_bins]

    def transform(self, state):
        transformed_state = np.zeros(state.shape)
        for i in range(len(state)):
            transformed_state[i] = np.digitize(state[i], self.state_feature_bins[i])

        transformed_state = int("".join(map(lambda x: str(int(x)), transformed_state)))
        return transformed_state
    
class Agent:
    def __init__(self, env):
        self.epsilon = 0.15
        self.gamma = 0.9
        self.alpha = 0.01

        self.n_states = 10**env.observation_space.shape[0]
        self.n_actions = env.action_space.n
        self.Q = np.random.rand(self.n_states, self.n_actions)

        self.transformer = StateTransformer()

    def act(self, state):
        state = self.transformer.transform(state)
        if np.random.random() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.Q[state])
        
    def update(self, state, action, reward, next_state, done):
        state = self.transformer.transform(state)
        next_state = self.transformer.transform(next_state)

        # Defining the target of our update rule
        if done:
            y = reward
        else:
            y = reward + self.gamma*np.max(self.Q[next_state])
        self.Q[state, action] += self.alpha*(y - self.Q[state, action])

def gather_samples(episodes):
    samples = []
    for i in range(episodes):
        state, _ = env.reset()
        samples.append(state)
        done = False
        steps = 0
        while not done and steps < 1000:
            action = env.action_space.sample()
            state, reward, done, _, _ = env.step(action)
            samples.append(state)
    return np.array(samples)

def play_one_episode(max_steps=10000, render=False, train=False):
    episode_reward = 0
    steps = 0
    state, _ = env.reset()

    done = False
    while not done and steps < max_steps:
        action = agent.act(state)
        next_state, reward, done, _, _ = env.step(action)
        episode_reward += reward
        steps += 1

        if render: env.render()
        if train: 
            if done: reward = -300
            agent.update(state, action, reward, next_state, done)

        state = next_state
    return episode_reward

if __name__ == '__main__':
    env = gym.make('CartPole-v1')

    '''
    state_samples = gather_samples(100)
    n_bins = 100
    
    figure, ax = plt.subplots(2,2)
    ax[0,0].hist(state_samples[:,0], bins=n_bins)
    ax[0,1].hist(state_samples[:,1], bins=n_bins)
    ax[1,0].hist(state_samples[:,2], bins=n_bins)
    ax[1,1].hist(state_samples[:,3], bins=n_bins)
    plt.show()
    '''

    agent = Agent(env)

    episodes_to_train = 10000
    rewards = np.zeros(episodes_to_train)

    for episode in range(episodes_to_train):
        print('Episode:', episode)
        episode_reward = play_one_episode(train=True)
        rewards[episode] = episode_reward
    
    env.close()

    plt.plot(rewards)
    plt.title('Rewards per episode')
    plt.show()

    env = gym.make('CartPole-v1', render_mode='human')
    agent.epsilon = 0 # Pure exploitation

    play_one_episode(render=True)
    play_one_episode(render=True)
    env.close()