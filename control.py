import numpy as np
import os
from env import make_env
import util
from logx import setup_logger_kwargs, EpochLogger

exp_name = "control"
seed = 1007
env_config = '1x1_mix_ms_una.json'
iterations = 1
max_ep_len = 600

wp = 0
wi = 0.1
wd = 0.1

initial_bound = 1.1
final_bound = 1.01
bound_decay = 0.98

env = make_env(util.ENV_CONFIG_DIR + env_config)

obs = []
actions = []
action_sign = np.array([-1, -1])
for i in range(iterations):
    current_bound = initial_bound
    o = env.reset()
    real_action = env.action_space.default() * 0.5
    for t in range(max_ep_len):
        o, r, d, _ = env.step(real_action)
        obs.append(o)
        actions.append(real_action)

        vp = o
        vi = np.mean(obs[-5:])
        vd = np.mean(np.diff(obs, axis=0)[-5:])
        vd = 0 if np.isnan(vd) else vd
        delta = np.exp((wp * vp + wi * vi + wd * vd) * action_sign)
        delta = np.clip(delta, 1. / current_bound, current_bound)
        #print(real_action, o, delta)
        real_action = env.action_space.clip(real_action * delta)
        current_bound = np.maximum(final_bound, current_bound * bound_decay)
    
print(np.mean(np.abs(obs[-20:])) * 100)
logger_kwargs = setup_logger_kwargs(exp_name, seed, data_dir=util.LOG_DIR+os.path.splitext(env_config)[0])
logger = EpochLogger(**logger_kwargs)
#util.plot_seq_obs_and_actions(np.abs(obs), actions, env.action_space.high, logger.output_dir + '/actions.png')