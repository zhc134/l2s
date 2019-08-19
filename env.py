from generator import Generator
from processor import Processor
import engine
import numpy as np
import json
import util

def make_env(env_config_file):
    return Env(env_config_file)

class ActionSpace():
    def __init__(self, high):
        self.shape = (len(high),)
        self.high = np.array(high)
    
    def sample(self):
        pi_a = np.random.random_sample(self.shape[0])
        return pi_a, pi_a * self.high
    
    def default(self):
        return self.high / 2

    def clip(self, a):
        return np.clip(a, 1e-3, self.high)


class ObservationSpace():
    def __init__(self, dim):
        self.shape = (dim,)
        self.default = np.zeros(dim)

class Env():
    def __init__(self, env_config_file):
        with open(env_config_file) as f:
            config = json.load(f)
        self.observation_space = ObservationSpace(config["observation_space"]["dim"])
        self.action_space = ActionSpace(config["action_space"]["high"])

        self.params_template = config["params_template"]
        self.target_params = config["target_params"]
        
        data_dir = util.SETTING_DIR + config["setting_dir"]
        roadnet_file = data_dir + config["roadnet_file"]
        flow_file = data_dir + config["flow_file"]
        signal_file = data_dir + config["signal_file"]
        self.observed_file = data_dir + config["observed_file"]
        self.f_observed = open(self.observed_file)
        self.gen = Generator(flow_file, signal_file)

        self.proc = Processor()
        self.eng = engine.Engine(1.0, 2, True, True, False)
        self.eng.load_roadnet(roadnet_file)

        self.t = 0
        self.total_reward = 0
        self.d = False
        self.steps = self.gen.steps

        self.reset()

    def reset(self):
        self.t = 0
        self.total_reward = 0
        self.eng.reset()
        self.gen.reset()
        self.state = self.observation_space.default
        self.d = False
        self.f_observed.seek(0)
        self._move_to_next_step()
        return self.state

    def get_observed(self):
        return self.f_observed.readline()

    def get_params(self, a):
        params = self.params_template.copy()
        for i, target_param in enumerate(self.target_params):
            params[target_param] = a[i]
        return params

    def _internal_step(self, params):
        self.gen.step(self.eng, params)
        self.eng.next_step()
        self.t += 1
        generated = self.proc.get_output(self.eng)
        observed = self.get_observed()
        self.state = self.proc.get_state(generated, observed)
        rew = self.proc.get_reward(generated, observed)
        if self.t >= self.steps:
            self.d = True
        return rew

    def _move_to_next_step(self):
        rews = []
        while not self.d:
            if self.gen.action_required():
                break
            rews.append(self._internal_step(None))
        return rews

    def step(self, a):
        params = self.get_params(a)
        rew = self._internal_step(params)
        rews = [rew]
        rews += self._move_to_next_step()
        rew = np.mean(rews)
        self.total_reward += rew
        return self.state, rew, self.d, dict()

    def __del__(self):
        self.f_observed.close()

    