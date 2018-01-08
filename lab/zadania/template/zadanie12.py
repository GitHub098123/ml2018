import numpy as np
from gym import spaces

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class LogisticAgent:

    def __init__(self, learning_rate=.1, seed=43):
        self.learning_rate = learning_rate
        self.rng = np.random.RandomState(seed=seed)
        self.w = None
        self.b = None

    def reset_params(self, observation_space, action_space):
        assert(isinstance(observation_space, spaces.Box))
        assert(isinstance(action_space, spaces.Discrete))
        assert(action_space.n == 2)
        self.w = self.rng.normal(size=observation_space.shape)
        self.b = 0.

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.w) + self.b)

    def actions(self, X):
        ...

    def action(self, x):
        return self.actions(np.array([x]))[0]

    def update(self, episode):
        ...

class AgentTrainer:

    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.agent.reset_params(env.observation_space, env.action_space)

    def episode(self, n_steps=200, break_if_done=True, show=False):
        ...

    def train(
            self, n_episodes, max_steps_per_episode,
            report_every_n_episodes, update_every_n_episodes,
            n_repeats_per_report):
        ep_buffer = []
        rewards = []
        for episode_idx in range(n_episodes):
            ep_buffer.append(self.episode(n_steps=max_steps_per_episode))
            rewards.append(ep_buffer[-1]["rewards"].sum())
            if ((episode_idx + 1) % update_every_n_episodes == 0):
                for episode in ep_buffer:
                    self.agent.update(episode)
                ep_buffer = []
            if ((episode_idx + 1) % report_every_n_episodes == 0):
                for _ in range(n_repeats_per_report):
                    episode = self.episode(show=True)
        return np.array(rewards)
