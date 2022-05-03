import numpy as np
import gym
from tqdm import tqdm


class MountainCarBaseAgent():
    def __init__(self, env, num_episodes, bin, min_lr, epsilon, lr,
                 discount_factor, decay):
        self.bin = bin
        self.num_episodes = num_episodes
        self.min_lr = min_lr
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.decay = decay
        self.learning_rate = lr

        self.env = env

        self.upper_bounds = self.env.observation_space.high
        self.lower_bounds = self.env.observation_space.low

        self.position_bins = np.linspace(self.lower_bounds[0], self.upper_bounds[0], num=self.bin)
        self.velocity_bins = np.linspace(self.lower_bounds[1], self.upper_bounds[1], num=self.bin)
        self.Q = np.zeros((self.bin, self.bin, self.env.action_space.n))

    def discretize_state(self, obs):
        discrete_pos = np.digitize(obs[0], bins=self.position_bins)
        discrete_vel = np.digitize(obs[1], bins=self.velocity_bins)
        discrete_state = np.array([discrete_pos, discrete_vel]).astype(np.int)
        return tuple(discrete_state)

    def choose_action(self, state, greedy=False):
        if not greedy:
            if np.random.random() < self.epsilon:
                return self.env.action_space.sample()
            else:
                return np.argmax(self.Q[state])
        else:
            return np.argmax(self.Q[state])

    def get_learning_rate(self):
        return max(self.min_lr, self.learning_rate - self.learning_rate * self.decay)

    def run(self):
        state = self.env.reset()
        total_reward = 0
        for ep in range(50000):
            state = self.discretize_state(state)
            action = self.choose_action(state, greedy=True)
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
            state = obs
        return total_reward

    def run_episodes(self, play_eps=500):
        stepsRecorder = []
        for _ in range(play_eps):
            stepsRecorder.append(self.run())
        stepsRecorder = np.array(stepsRecorder)
        print(f'Finish with mean steps: {np.mean(stepsRecorder)} in {play_eps} episodes')


class MountainCarTDnAgent(MountainCarBaseAgent):
    def __init__(self, env, n, bin=20, num_episodes=1000, discount_factor=0.95, min_lr=0.1, lr=1.0,
                 decay=0.25, epsilon=0.2):
        super().__init__(env, num_episodes, bin, min_lr, epsilon, lr,
                         discount_factor, decay)
        self.n = n
        self.state_store = {}
        self.action_store = {}
        self.reward_store = {}

    def train(self):

        for ep in tqdm(range(self.num_episodes)):
            T = np.inf
            tau = 0
            t = -1

            state = self.env.reset()

            state = self.discretize_state(state)
            action = self.choose_action(state)
            self.state_store[0] = state
            self.action_store[0] = action

            self.learning_rate = self.get_learning_rate()

            while tau < (T - 1):
                t += 1
                if t < T:
                    state, reward, done, _ = self.env.step(action)
                    state = self.discretize_state(state)
                    self.state_store[(t + 1) % (self.n + 1)] = state
                    self.reward_store[(t + 1) % (self.n + 1)] = reward

                    if done:
                        T = t + 1
                    else:
                        action = self.choose_action(state)
                        self.action_store[(t + 1) % (self.n + 1)] = action
                tau = t - self.n + 1

                if tau >= 0:
                    G = np.sum([self.discount_factor ** (i - tau - 1) * self.reward_store[i % (self.n + 1)] for i in
                                range(tau + 1, min(tau + self.n, T) + 1)])
                    if tau + self.n < T:
                        state_tau = self.state_store[(tau + self.n) % (self.n + 1)]
                        action_tau = self.action_store[(tau + self.n) % (self.n + 1)]
                        G += (self.discount_factor ** self.n) * self.Q[state_tau][action_tau]
                    state_tau, action_tau = self.state_store[tau % (self.n + 1)], self.action_store[tau % (self.n + 1)]
                    self.Q[state_tau][action_tau] += self.learning_rate * (G - self.Q[state_tau][action_tau])


env = gym.make('MountainCar-v0')
# hyperparameters
bin = 20
num_episodes = 5000
min_lr = 0.1
epsilon = 0.2
lr = 1.0
discount_factor = 0.95
lr_decay = 0.25

print('train using n=4: ')
env.reset()
agent = MountainCarTDnAgent(env, n=4,
                            num_episodes=num_episodes,
                            min_lr=min_lr, epsilon=epsilon,
                            lr=lr, discount_factor=discount_factor,
                            decay=lr_decay)
agent.train()
agent.run_episodes()
