import numpy as np
import json
from generator import Generator
from processor import Processor
import engine
import os

class GMSimAgent():
    def __init__(self, log=True, ground_truth_suffix=None, dir="data/setting/1x1/"):
        if log and ground_truth_suffix is None:
            raise Exception("no ground_truth_suffix provided!!")

        self.log = log

        self.params_template = {
            "length": 5.0,
            "width": 2.0,
            "maxPosAcc": 3.0,
            "maxNegAcc": 4.5,
            "usualPosAcc": 2.0,
            "usualNegAcc": 2.5,
            "minGap": 2.5,
            "maxSpeed": 15,
            "headwayTime": 1.5
        }
        
        data_dir = "data/setting/1x1/"
        with open(os.path.join(dir, "config.json")) as f:
            config = json.load(f)
            self.target_params = config["target_params"]
            self.probs = config["probs"]
            self.means = config["means"]
            self.stds = config["stds"]

        self.suffix = "_1_1"
        roadnet_file = data_dir + 'roadnet%s.json' % self.suffix
        flow_file = data_dir + 'flow%s.jsonl' % self.suffix
        signal_file = data_dir + 'signal%s.jsonl' % self.suffix
        self.gen = Generator(flow_file, signal_file)

        self.proc = Processor()
        self.eng = engine.Engine(1.0, 2, True, True, False)
        self.eng.load_roadnet(roadnet_file)
        
        self.t = 0
        if self.log:
            self.observed_file = data_dir + 'observed{}_{}.txt'.format(self.suffix, ground_truth_suffix)
            self.f_observed = open(self.observed_file, "w")

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
        self.eng.print_log('data/frontend/web/roadnet%s.json' % self.suffix, 'data/frontend/web/replay%s.txt' % self.suffix)

    def __del__(self):
        if self.log:
            self.f_observed.close()

if __name__ == '__main__':
    import os
    import sys

    np.random.seed(1007)
    fsa = GMSimAgent(log=True, ground_truth_suffix=sys.argv[1])
    for t in range(fsa.gen.steps):
        if t % 100 == 0:
            print(t)
        fsa.next_step()
    fsa.print_log()
