import numpy as np
import json
from generator import Generator
from processor import Processor
import engine
import os
import util

class SimAgent():
    def __init__(self, env_config):
        config = json.load(open(util.ENV_CONFIG_DIR + env_config))
        self.params_template = config["params_template"]
        self.target_params = config["target_params"]
        
        sim_config = config["sim_config"]
        self.probs = sim_config["probs"]
        self.means = sim_config["means"]
        self.stds = sim_config["stds"]
        np.random.seed(sim_config["seed"])
        
        self.setting_dir = util.SETTING_DIR + config["setting_dir"]
        roadnet_file = self.setting_dir + config["roadnet_file"]
        flow_file = self.setting_dir + config["flow_file"]
        signal_file = self.setting_dir + config["signal_file"]
        self.observed_file = self.setting_dir + config["observed_file"]
        self.log = True
        self.f_observed = open(self.observed_file, "w")
        self.gen = Generator(flow_file, signal_file)

        self.proc = Processor()
        self.eng = engine.Engine(1.0, 2, True, True, False)
        self.eng.load_roadnet(roadnet_file)
        
        self.t = 0

    def set_params(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def generate_observed(self):
        print('generating...')
        for _ in range(self.gen.steps):
            self.next_step()
        print('generated to %s' % self.observed_file)

    def get_params(self):
        params = self.params_template.copy()
        k = np.random.choice(len(self.probs), p = self.probs)
        for i, target_param in enumerate(self.target_params):
            params[target_param] = np.random.normal(self.means[k][i], self.stds[i])
        return params

    def next_step(self):
        self.gen.step(self.eng, self.get_params())
        self.eng.next_step()
        self.t += 1
        ret = self.proc.get_output(self.eng, self.log)

        if self.log:
            self.f_observed.write(ret + '\n')
        else:
            return ret

    def reset(self):
        self.gen.reset()
        self.eng.reset()
        if self.log:
            self.f_observed.close()
            self.f_observed = open(self.observed_file, "w")
        self.t = 0

    def print_log(self):
        self.eng.print_log('data/frontend/web/roadnet.json', 'data/frontend/web/replay.txt')

    def __del__(self):
        if self.log:
            self.f_observed.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_config', type=str)
    args = parser.parse_args()

    sa = SimAgent(args.env_config)
    sa.generate_observed()
    sa.print_log()
