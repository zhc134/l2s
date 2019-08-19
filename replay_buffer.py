import numpy as np
import heapq


def get_replay_buffer(rb_type):
    if rb_type == "FIFO":
        return FIFOReplayBuffer
    else:
        if rb_type == "Random":
            return RandomReplayBuffer
        elif rb_type == "GDM":
            return GDMReplayBuffer
        elif rb_type == "Surprise":
            return SurpriseReplayBuffer
        elif rb_type == "Reward":
            return RewardReplayBuffer
        elif rb_type == "CM":
            return CMReplayBuffer
        else:
            raise Exception("wrong replay buffer type!")


class ReplayBuffer():
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        raise NotImplementedError()

    def sample_batch(self, batch_size=32):
        raise NotImplementedError()


class FIFOReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size):
        super().__init__(obs_dim, act_dim, size)

    def store(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.size = min(self.size + 1, self.max_size)
        self.ptr = (self.ptr + 1) % self.max_size

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class RandomReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, next_obs, done):
        if self.size < self.max_size:
            self.size = self.size + 1
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr + 1)
        else:
            self.ptr = np.random.randint(self.size)
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class GDMReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size, rate):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.rate = rate
        self.num = 0
        self.FIFO = int(np.round(self.rate * self.max_size))

    def store(self, obs, act, rew, next_obs, done):
        self.num = self.num + 1
        if self.size < self.max_size:
            self.size = self.size + 1
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.ptr = (self.ptr + 1)
        else:
            if np.random.rand(1, 1) > self.rate:
                if np.random.rand(1, 1) < ((self.max_size - self.FIFO) / (self.num + 0.0)):
                    ptr = np.random.randint(self.max_size - self.FIFO) + self.FIFO
                    self.obs1_buf[ptr] = obs
                    self.obs2_buf[ptr] = next_obs
                    self.acts_buf[ptr] = act
                    self.rews_buf[ptr] = rew
                    self.done_buf[ptr] = done
            else:
                self.ptr = min(self.ptr, self.FIFO - 1)
                self.obs1_buf[self.ptr] = obs
                self.obs2_buf[self.ptr] = next_obs
                self.acts_buf[self.ptr] = act
                self.rews_buf[self.ptr] = rew
                self.done_buf[self.ptr] = done
                self.ptr = (self.ptr + 1) % self.FIFO

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=self.obs1_buf[idxs],
                    obs2=self.obs2_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    done=self.done_buf[idxs])


class heapItem:
    def __init__(self, TD, ptr):
        self.TD = TD
        self.ptr = ptr

    def __lt__(self, other):
        if self.TD > other.TD:
            return -1
        elif self.TD < other.TD:
            return 1
        else:
            return 0


class SurpriseReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size):
        self.heap = []
        self.TDs_buf = []
        self.FIFO = FIFOReplayBuffer(obs_dim, act_dim, int(size * 0.3))
        super().__init__(obs_dim, act_dim, size - int(size * 0.3))

    def store2(self, obs, act, rew, next_obs, done, q, v):
        if np.random.rand(1, 1) < 0.3:
            self.FIFO.store(obs, act, rew, next_obs, done)
            return
        if self.size < self.max_size:
            self.size = self.size + 1
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.TDs_buf.append(heapItem(abs(rew + v - q), self.ptr))
            heapq.heappush(self.heap, self.TDs_buf[self.ptr])
            self.ptr = self.ptr + 1
        else:
            self.ptr = self.heap[0].ptr
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.TDs_buf[self.ptr] = heapItem(abs(rew + v - q), self.ptr)
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, self.TDs_buf[self.ptr])

    def sample_batch(self, batch_size=32):
        idxs1 = np.random.randint(0, int(self.size * 0.3), size=int(batch_size * 0.3))
        idxs2 = np.random.randint(0, self.size - int(self.size * 0.3), size=batch_size - int(batch_size * 0.3))
        return dict(obs1=np.concatenate((self.FIFO.obs1_buf[idxs1], self.obs1_buf[idxs2]), axis=0),
                    obs2=np.concatenate((self.FIFO.obs2_buf[idxs1], self.obs2_buf[idxs2]), axis=0),
                    acts=np.concatenate((self.FIFO.acts_buf[idxs1], self.acts_buf[idxs2]), axis=0),
                    rews=np.concatenate((self.FIFO.rews_buf[idxs1], self.rews_buf[idxs2]), axis=0),
                    done=np.concatenate((self.FIFO.done_buf[idxs1], self.done_buf[idxs2]), axis=0)
                    )


class RewardReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size):
        self.heap = []
        self.TDs_buf = []
        self.FIFO = FIFOReplayBuffer(obs_dim, act_dim, int(size * 0.3))
        super().__init__(obs_dim, act_dim, size - int(size * 0.3))

    def store2(self, obs, act, rew, next_obs, done, q, v):
        if np.random.rand(1, 1) < 0.3:
            self.FIFO.store(obs, act, rew, next_obs, done)
            return
        if self.size < self.max_size:
            self.size = self.size + 1
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.TDs_buf.append(heapItem(abs(rew), self.ptr))
            heapq.heappush(self.heap, self.TDs_buf[self.ptr])
            self.ptr = self.ptr + 1
        else:
            self.ptr = self.heap[0].ptr
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.TDs_buf[self.ptr] = heapItem(abs(rew), self.ptr)
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, self.TDs_buf[self.ptr])

    def sample_batch(self, batch_size=32):
        idxs1 = np.random.randint(0, int(self.size * 0.3), size=int(batch_size * 0.3))
        idxs2 = np.random.randint(0, self.size - int(self.size * 0.3), size=batch_size - int(batch_size * 0.3))
        return dict(obs1=np.concatenate((self.FIFO.obs1_buf[idxs1], self.obs1_buf[idxs2]), axis=0),
                    obs2=np.concatenate((self.FIFO.obs2_buf[idxs1], self.obs2_buf[idxs2]), axis=0),
                    acts=np.concatenate((self.FIFO.acts_buf[idxs1], self.acts_buf[idxs2]), axis=0),
                    rews=np.concatenate((self.FIFO.rews_buf[idxs1], self.rews_buf[idxs2]), axis=0),
                    done=np.concatenate((self.FIFO.done_buf[idxs1], self.done_buf[idxs2]), axis=0)
                    )


def MSE(target, prediction):
    if type(target) == np.float32:
        return (target - prediction) ** 2
    else:
        n = len(target)
    error = []
    for i in range(n):
        error.append(target[i] - prediction[i])
    squaredError = []
    for val in error:
        squaredError.append(val * val)
    return sum(squaredError) / len(squaredError)


class CMReplayBuffer(ReplayBuffer):
    def __init__(self, obs_dim, act_dim, size):
        self.heap = []
        self.TDs_buf = []
        self.FIFO = FIFOReplayBuffer(obs_dim, act_dim, int(size * 0.3))
        super().__init__(obs_dim, act_dim, size - int(size * 0.3))

    def getRank(self):
        rank = 0.0
        for i in range(self.size):
            rank = rank + MSE(target=self.obs1_buf[self.ptr], prediction=self.obs1_buf[i]) + \
                   MSE(target=self.obs2_buf[self.ptr], prediction=self.obs2_buf[i]) + \
                   MSE(target=self.acts_buf[self.ptr], prediction=self.acts_buf[i]) + \
                   MSE(target=self.rews_buf[self.ptr], prediction=self.rews_buf[i])
        return  rank / self.size

    def store(self, obs, act, rew, next_obs, done):
        if np.random.rand(1, 1) < 0.3:
            self.FIFO.store(obs, act, rew, next_obs, done)
            return
        if self.size < self.max_size:
            self.size = self.size + 1
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.TDs_buf.append(heapItem(self.getRank(), self.ptr))
            heapq.heappush(self.heap, self.TDs_buf[self.ptr])
            self.ptr = self.ptr + 1
        else:
            self.ptr = self.heap[0].ptr
            self.obs1_buf[self.ptr] = obs
            self.obs2_buf[self.ptr] = next_obs
            self.acts_buf[self.ptr] = act
            self.rews_buf[self.ptr] = rew
            self.done_buf[self.ptr] = done
            self.TDs_buf[self.ptr] = heapItem(self.getRank(), self.ptr)
            heapq.heappop(self.heap)
            heapq.heappush(self.heap, self.TDs_buf[self.ptr])

    def sample_batch(self, batch_size=32):
        idxs1 = np.random.randint(0, int(self.size * 0.3), size=int(batch_size * 0.3))
        idxs2 = np.random.randint(0, self.size - int(self.size * 0.3), size=batch_size - int(batch_size * 0.3))
        return dict(obs1=np.concatenate((self.FIFO.obs1_buf[idxs1], self.obs1_buf[idxs2]), axis=0),
                    obs2=np.concatenate((self.FIFO.obs2_buf[idxs1], self.obs2_buf[idxs2]), axis=0),
                    acts=np.concatenate((self.FIFO.acts_buf[idxs1], self.acts_buf[idxs2]), axis=0),
                    rews=np.concatenate((self.FIFO.rews_buf[idxs1], self.rews_buf[idxs2]), axis=0),
                    done=np.concatenate((self.FIFO.done_buf[idxs1], self.done_buf[idxs2]), axis=0)
                    )
