import numpy as np

# Expects tuples of (id, state, next_state, action, possible_action, suction, reward, done)
class ReplayBuffer(object):
    def __init__(self, max_size=1e5):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def add(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        ids, states, next_states, actions, p_actions, suctions, rewards, dones = [], [], [], [], [], [], [], []
        # (id, state, next_state, action, possible_action, suction, reward, done)

        for i in ind:
            id, state, next_state, action, p_action, suction, reward, done = self.storage[i]
            ids.append(np.array(id, copy=False))
            states.append(np.array(state, copy=False))
            next_states.append(np.array(next_state, copy=False))
            actions.append(np.array(action, copy=False))
            p_actions.append(np.array(p_action, copy=False))
            suctions.append(np.array(suction, copy=False))
            rewards.append(np.array(reward, copy=False))
            dones.append(np.array(done, copy=False))

        return [np.array(ids), np.array(states), np.array(next_states),
                [np.array(actions), np.array(p_actions), np.array(suctions)],
                np.array(rewards).reshape(-1, 1), np.array(dones).reshape(-1, 1)]