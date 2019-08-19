import numpy as np
import tensorflow as tf
import time
import core
from core import get_ppo_actor_critic
from logx import EpochLogger
from env import make_env
import util

ALG_NAME = "ppo"

class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self):
        path_slice = slice(self.path_start_idx, self.ptr)
        weights = np.array([1])
        self.ret_buf[path_slice] = core.cum_weighted_sum(self.rew_buf[path_slice], weights)
        self.adv_buf[path_slice] = np.sum(np.abs(self.obs_buf[path_slice]), axis=1) - np.abs(self.rew_buf[path_slice])
        self.path_start_idx = self.ptr

    def get(self):
        s = np.ones(self.max_size, dtype=bool)
        s[self.ptr:] = False
        self.ptr, self.path_start_idx = 0, 0
        adv_mean = np.mean(self.adv_buf[s])
        adv_std = np.std(self.adv_buf[s])
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf[s], self.act_buf[s], self.adv_buf[s], self.logp_buf[s], self.rew_buf[s]]


"""

Proximal Policy Optimization (by clipping), 

with early stopping based on approximate KL

"""
def ppo(
        env_config, ac_type, ac_kwargs, clip_ratio,
        epochs, steps_per_epoch, optimizer, lr, train_pi_iters, max_ep_len, target_kl,
        logger_kwargs, seed
    ):
    logger = EpochLogger(**logger_kwargs)
    configs = locals().copy()
    configs.pop("logger")
    logger.save_config(configs)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = make_env(env_config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_high = env.action_space.high
    
    obs_ph, a_ph, adv_ph, logp_old_ph = core.placeholders(obs_dim, act_dim, None, None)
    all_phs = [obs_ph, a_ph, adv_ph, logp_old_ph]

    actor_critic = get_ppo_actor_critic(ac_type)
    pi, logp, logp_pi = actor_critic(obs_ph, a_ph, **ac_kwargs)
    
    # Experience buffer
    buf = PPOBuffer(obs_dim, act_dim, steps_per_epoch)

    # PPO objectives
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))

    # Optimizers
    if optimizer == "adam":
        train_pi = tf.train.AdamOptimizer(learning_rate=lr).minimize(pi_loss)
    elif optimizer == "sgd":
        train_pi = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(pi_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def update():
        
        print(sess.run(tf.trainable_variables()))

        data = buf.get()
        #util.plot_adv(data[0] * act_high, data[1], logger.output_dir + "/ep_adv%s.png" % epoch)
        inputs = {k:v for k,v in zip(all_phs, data[:4])}
        pi_l_old, ent = sess.run([pi_loss, approx_ent], feed_dict=inputs)

        # Training
        for i in range(train_pi_iters):
            _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
            if kl > 1.5 * target_kl:
                logger.log('Early stopping at step %d due to reaching max kl.' % i)
                break
        logger.store(StopIter=i)

        # Log changes from update
        pi_l_new, kl, cf = sess.run([pi_loss, approx_kl, clipfrac], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, 
                     KL=kl, Entropy=ent, ClipFrac=cf,
                     DeltaLossPi=(pi_l_new - pi_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    real_action = env.action_space.default()
    o, r, d, _ = env.step(real_action)

    episode_actions = []
    episode_obs = []
    episode_actions.append(real_action)
    episode_obs.append(o)

    print(tf.trainable_variables())
    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        episode_count = 0
        ep_actions = []
        for t in range(steps_per_epoch):
            a, logp_t = sess.run([pi, logp_pi], feed_dict={obs_ph: o.reshape(1, -1)})
            delta = np.exp(a[0])
            delta = np.clip(delta, 0.95, 1.05)
            real_action = env.action_space.clip(real_action * delta)

            o, r, d, _ = env.step(real_action)
            
            buf.store(o, a, r, logp_t)

            ep_actions.append(real_action)
            episode_actions.append(real_action)
            episode_obs.append(o)
            ep_ret += r
            ep_len += 1

            if ep_len == max_ep_len or t == steps_per_epoch-1:
                buf.finish_path()
                logger.store(EpRet=ep_ret, EpLen=ep_len)
                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                real_action = env.action_space.default()
                o, r, d, _ = env.step(real_action)

                util.plot_seq_obs_and_actions(episode_obs, episode_actions, act_high, logger.output_dir + '/episode_actions_%d_%d.png' % (epoch, episode_count))
                episode_count += 1
                episode_actions = []
                episode_obs = []
                episode_actions.append(real_action)
                episode_obs.append(o)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('ClipFrac', average_only=True)
        logger.log_tabular('StopIter', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

        util.plot_actions(ep_actions, act_high, logger.output_dir + '/ep_actions%d.png' % epoch)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('model_config', type=str)
    parser.add_argument('env_config', type=str)
    parser.add_argument('seed', type=int, default=1007)
    args = parser.parse_args()

    import json
    import os
    model_config = json.load(open(util.MODEL_CONFIG_DIR + ALG_NAME + "/" + args.model_config))

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=util.LOG_DIR+os.path.splitext(args.env_config)[0])

    ppo(
        env_config=util.ENV_CONFIG_DIR+args.env_config,
        ac_type=model_config["ac_type"],
        ac_kwargs=model_config["ac_kwargs"],
        clip_ratio=model_config["clip_ratio"],
        epochs=model_config["epochs"],
        steps_per_epoch=model_config["steps_per_epoch"],
        optimizer=model_config["optimizer"],
        lr=model_config["lr"],
        train_pi_iters=model_config["train_pi_iters"],
        max_ep_len=model_config["max_ep_len"],
        target_kl=model_config["target_kl"],
        logger_kwargs=logger_kwargs,
        seed=args.seed
    )