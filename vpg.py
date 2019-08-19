import numpy as np
import tensorflow as tf
import time
import scipy

import core
from logx import EpochLogger
from env import make_env
import util

ALG_NAME = "vpg"

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

class VPGBuffer:
    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((size, act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, obs, act, rew, val, logp):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.val_buf[self.ptr] = val
        self.logp_buf[self.ptr] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(deltas, self.gamma * self.lam)
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]
        #print(self.ret_buf[path_slice])

        self.path_start_idx = self.ptr

    def get(self):
        assert self.ptr == self.max_size
        self.ptr, self.path_start_idx = 0, 0
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.obs_buf, self.act_buf, self.adv_buf, self.ret_buf, self.logp_buf]

def gaussian_mlp_actor_critic(x, a):
    with tf.variable_scope('pi'):
        act_dim = a.shape.as_list()[-1]
        mu = core.mlp(x, [act_dim], use_bias=False)
        log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
        std = tf.exp(log_std)
        pi = mu + tf.random_normal(tf.shape(mu)) * std
        logp = core.gaussian_likelihood(a, mu, log_std)
        logp_pi = core.gaussian_likelihood(pi, mu, log_std)
    with tf.variable_scope('v'):
        v = tf.squeeze(core.mlp(tf.abs(x), [1]), axis=1)
    return pi, logp, logp_pi, v


def vpg(
        env_config, ac_type, ac_kwargs, gamma, lam,
        epochs, steps_per_epoch, lr, train_v_iters, max_ep_len,
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
    
    obs_ph, a_ph, adv_ph, ret_ph, logp_old_ph = core.placeholders(obs_dim, act_dim, None, None, None)

    actor_critic = gaussian_mlp_actor_critic
    pi, logp, logp_pi, v = actor_critic(obs_ph, a_ph, **ac_kwargs)

    all_phs = [obs_ph, a_ph, adv_ph, ret_ph, logp_old_ph]
    get_action_ops = [pi, v, logp_pi]
    
    # Experience buffer
    buf = VPGBuffer(obs_dim, act_dim, steps_per_epoch, gamma, lam)

    # VPG objectives
    pi_loss = -tf.reduce_mean(logp * adv_ph)
    v_loss = tf.reduce_mean((ret_ph - v)**2)

    # Info (useful to watch during learning)
    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute

    # Optimizers
    train_pi = tf.train.AdamOptimizer(learning_rate=lr).minimize(pi_loss)
    train_v = tf.train.AdamOptimizer(learning_rate=lr).minimize(v_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def update():
        buffer_data = buf.get()
        #util.plot_adv(data[0] * act_high, data[1], logger.output_dir + "/ep_adv%s.png" % epoch)
        inputs = {k:v for k,v in zip(all_phs, buffer_data)}
        pi_l_old, v_l_old, ent = sess.run([pi_loss, v_loss, approx_ent], feed_dict=inputs)

        sess.run(train_pi, feed_dict=inputs)

        # Training
        for _ in range(train_v_iters):
            sess.run(train_v, feed_dict=inputs)

        # Log changes from update
        pi_l_new, v_l_new, kl, v_new = sess.run([pi_loss, v_loss, approx_kl, v], feed_dict=inputs)
        logger.store(LossPi=pi_l_old, LossV=v_l_old,
                     KL=kl, Entropy=ent,
                     DeltaLossPi=(pi_l_new - pi_l_old),
                     DeltaLossV=(v_l_new - v_l_old))

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    real_action = env.action_space.default()

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(steps_per_epoch):
            a, v_t, logp_t = sess.run(get_action_ops, feed_dict={obs_ph: o.reshape(1, -1)})

            buf.store(o, a, r, v_t, logp_t)
            logger.store(VVals=v_t)

            delta = np.exp(a[0])
            delta = np.clip(delta, 0.9, 1.1)
            real_action = env.action_space.clip(real_action * delta)

            o, r, d, _ = env.step(real_action)
            ep_ret += r
            ep_len += 1

            if ep_len == max_ep_len or t == steps_per_epoch-1:
                last_val = sess.run(v, feed_dict={obs_ph: o.reshape(1, -1)})
                #print(last_val)
                buf.finish_path(last_val)
                logger.store(EpRet=ep_ret, EpLen=ep_len)

                o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
                real_action = env.action_space.default()

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet', with_min_and_max=True)
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('VVals', with_min_and_max=True)
        logger.log_tabular('LossPi', average_only=True)
        logger.log_tabular('LossV', average_only=True)
        logger.log_tabular('DeltaLossPi', average_only=True)
        logger.log_tabular('DeltaLossV', average_only=True)
        logger.log_tabular('Entropy', average_only=True)
        logger.log_tabular('KL', average_only=True)
        logger.log_tabular('Time', time.time()-start_time)
        logger.dump_tabular()

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

    vpg(
        env_config=util.ENV_CONFIG_DIR+args.env_config,
        ac_type=model_config["ac_type"],
        ac_kwargs=model_config["ac_kwargs"],
        gamma=model_config["gamma"],
        lam=model_config["lam"],
        epochs=model_config["epochs"],
        steps_per_epoch=model_config["steps_per_epoch"],
        lr=model_config["lr"],
        train_v_iters=model_config["train_pi_iters"],
        max_ep_len=model_config["max_ep_len"],
        logger_kwargs=logger_kwargs,
        seed=args.seed
    )