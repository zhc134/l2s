from generator import Generator
from processor import Processor
import engine

import numpy as np
from itertools import product
import json

import util

class GridSearchAgent():
    def __init__(self, env_config, std, verbose, agent=None):
        self.std = std
        self.verbose = verbose
        self.agent = agent
        all_grid_search_params = {
            "maxPosAcc"  : [1.5, 7.5, 0.3],
            "maxNegAcc"  : [2.0, 6.0, 0.1],
            "usualNegAcc": [1.5, 3.5, 0.2],
            "minGap"     : [1.0, 5.0, 0.2],
            "maxSpeed"   : [12.8, 13.3, 0.1],
        }

        config = json.load(open(env_config))

        self.params = config["params_template"]
        for k in self.params:
            self.params[k] = [self.params[k]]
        self.target_params = config["target_params"]
        grid_search_params = [all_grid_search_params[target] for target in self.target_params]
        for i, target in enumerate(config["target_params"]):
            low, high, step = grid_search_params[i]
            self.params[target] = np.arange(low, high, step)
        
        data_dir = util.SETTING_DIR + config["setting_dir"]
        roadnet_file = data_dir + config["roadnet_file"]
        flow_file = data_dir + config["flow_file"]
        signal_file = data_dir + config["signal_file"]
        if not self.agent:
            self.observed_file = data_dir + config["observed_file"]
            self.f_observed = open(self.observed_file)
        self.gen = Generator(flow_file, signal_file)

        self.proc = Processor()
        self.eng = engine.Engine(1.0, 2, True, True, False)
        self.eng.load_roadnet(roadnet_file)

        self.steps = self.gen.steps

    def get_observed(self):
        return self.f_observed.readline()

    def reset(self):
        self.eng.reset()
        self.gen.reset()
        if not self.agent:
            self.f_observed.seek(0)

    def get_params(self, params_template):
        params = params_template.copy()
        for target_param in self.target_params:
            params[target_param] = np.random.normal(params[target_param], self.std)
        return params

    def run_single(self, params, times=3):
        rewards = [[] for i in range(times)]
        for i in range(times):
            self.reset()
            if self.agent:
                agent.reset()
            for _ in range(self.steps):
                self.gen.step(self.eng, self.get_params(params))
                self.eng.next_step()

                generated = self.proc.get_output(self.eng)
                observed = agent.next_step() if self.agent else self.get_observed()

                # get reward
                r = self.proc.get_reward(generated, observed)
                rewards[i].append(r)
            
        return np.mean(rewards)

    def run(self):
        best_r = None
        for params in iter(self):
            if self.verbose:
                print("Running Params\n%s" % params)
            r = self.run_single(params)
            if self.verbose:
                print("Reward %f\n==========" % r)
            if best_r is None or r > best_r:
                best_r = r
                best_params = params
        print("Best Reward %f\nBest Params:\n%s" % (best_r, best_params))
        return best_r

    def __iter__(self):
        keys, values = zip(*self.params.items())
        for v in product(*values):
            yield dict(zip(keys, v))

def test(dir):
    final_rewards = []
    for _ in range(5):
        agent1 = GMSimAgent(log=False, dir=dir)
        agent2 = GMSimAgent(log=False, dir=dir)
        proc = Processor()
        rewards = []
        for i in range(1200):
            r = proc.get_reward(agent1.next_step(), agent2.next_step())
            rewards.append(r)
        final_rewards.append(np.mean(rewards))
    print("Test Reward %f with std %f" % (np.mean(final_rewards), np.std(final_rewards)))
    return np.mean(final_rewards)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env')
    parser.add_argument('--std', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=1007)
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--agent', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--log', action='store_true')
    parser.add_argument('--dir', type=str, default='.')
    args = parser.parse_args()

    import os
    np.random.seed(args.seed)
    from gm_sim_agent import GMSimAgent
    agent = GMSimAgent(log=False, dir=args.dir) if args.agent else None
    
    gsa = GridSearchAgent(util.ENV_CONFIG_DIR + args.env, args.std, args.verbose, agent=agent)
    grid_r = gsa.run()
    if args.test:
        test_r = test(dir=args.dir)
        relative_r = - (test_r - grid_r) / grid_r
        print("loss reward is %f" % (relative_r))
        if args.log:
            with open(os.path.join(args.dir, "final.txt"), "w") as f:
                f.write(' '.join([str(relative_r), str(test_r), str(grid_r)]))