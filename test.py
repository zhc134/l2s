from fix_sim_agent import FixSimAgent
from gm_sim_agent import GMSimAgent
from processor import Processor
import numpy as np

if __name__ == '__main__':
    import os

    final_rewards = []
    for _ in range(5):
        # agent1 = FixSimAgent(log=False, std=0.1)
        agent1 = GMSimAgent(log=False)
        # agent2 = FixSimAgent(log=False)
        agent2 = GMSimAgent(log=False)
        proc = Processor()
        rewards = []
        for i in range(3600):
            r = proc.get_reward(agent1.next_step(), agent2.next_step())
            rewards.append(r)
            #print(r)
        print(np.mean(rewards))
        final_rewards.append(np.mean(rewards))
    print(np.mean(final_rewards), np.std(final_rewards))

