from data_generator import DataGenerator
from fix_sim_agent import FixSimAgent
from reward import Reward

import numpy as np
from itertools import product
import engine
from random import Random
from sklearn.neighbors import KernelDensity
from matplotlib import pyplot as plt
from datetime import datetime



class ABCAgent():
    def __init__(self):
        SEED = datetime.now()
        SEED = 0
        
        self.n = 100
        self.tolerance = 1e-2
        self.bandwidth = 0.3

        self.maxNegAccMin = 3.0
        self.maxNegAccMax = 6.0

        self.run_cnt = 3600

        self.params_template = {
            "length": 5.0,
            "width": 2.0,
            "maxPosAcc": 3.0,
            "maxNegAcc": 4.5,
            "usualPosAcc": 2.0,
            "usualNegAcc": 2.5,
            "minGap": 2.5,
            "maxSpeed": 50/3,
            "headwayTime": 1.5
        }

        self.memo = "1_3"
        roadnet_file = 'data/roadnet/roadnet_{0}.json'.format(self.memo)
        flow_file = 'data/flow/flow_{0}.json'.format(self.memo)
        signal_file = 'data/signal/signal_{0}.json'.format(self.memo)

        self.observed_file = "data/gmsa_observed_%s.txt" % self.memo

        self.rand = Random(SEED)
        self.accepted = []
        self.kde = KernelDensity(self.bandwidth)
        
        self.reward = Reward()
        self.gen = DataGenerator(flow_file, signal_file)
        self.eng = engine.Engine(1.0, 8, True, True)
        self.eng.load_roadnet(roadnet_file)
        
        self.f = open(self.observed_file)

    def get_params(self):
        params = self.params_template.copy()
        params["maxNegAcc"] = self.maxNegAccMin + self.rand.random() * (self.maxNegAccMax - self.maxNegAccMin)
        return params

    def get_observed(self):
        return self.f.readline()

    def run_single(self, params):
        self.f = open(self.observed_file)
        self.eng.reset()
        losses = []
        for t in range(self.run_cnt):
            self.gen.simulate_data(self.eng, params, t)

            self.eng.next_step()

            generated = self.reward.get_output(self.eng)
            observed = self.get_observed()

            # get loss
            loss = -self.reward.process(generated, observed)["lane_speed"]
            losses.append(loss)

        return np.mean(losses)

    def plot_density(self, fname):
        x = np.linspace(self.maxNegAccMin, self.maxNegAccMax, 100)
        plt.plot(x, np.exp(self.kde.score_samples(x[:, np.newaxis])))
        plt.savefig(fname)

    def run(self):
        for _ in range(self.n):
            params = self.get_params()
            print("MaxNegAcc %.6f: " % params["maxNegAcc"], end="")
            loss = self.run_single(params)
            print("Loss %f" % loss)
            if loss < self.tolerance:
                self.accepted.append(params["maxNegAcc"])
        print(self.accepted)
        self.kde.fit(np.array(self.accepted).reshape((-1, 1)))
        self.plot_density('data/kde.png')
                
if __name__ == '__main__':
    abca = ABCAgent()
    abca.run()
