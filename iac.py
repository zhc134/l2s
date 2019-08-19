import numpy as np
import tensorflow as tf
import time
import os

import core
from core import get_vars
from logx import EpochLogger
from env import make_env
from replay_buffer import get_replay_buffer
import util


def iac(
        env_config, ac_type, ac_kwargs, rb_type, rb_kwargs, 
        gamma, lr, polyak, batch_size, 
        epochs, start_steps, steps_per_epoch, inc_ep,
        max_ep_len, test_max_ep_len, number_of_tests_per_epoch, q_pi_sample_size, z_dim, z_type, act_noise, test_without_state,
        logger_kwargs, seed
    ):
    logger = EpochLogger(**logger_kwargs)
    configs = locals().copy()
    configs.pop("logger")
    logger.save_config(configs)

    tf.set_random_seed(seed)
    np.random.seed(seed)

    env, test_env = make_env(env_config), make_env(env_config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    act_high = env.action_space.high

    # Inputs to computation graph
    x_ph, a_ph, z_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, z_dim, obs_dim, None, None)

    actor_critic = core.get_iac_actor_critic(ac_type)
    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q1, q2, q1_pi, q2_pi, v = actor_critic(x_ph, a_ph, z_ph, **ac_kwargs)
    
    # Target networks
    with tf.variable_scope('target'):
        _, _, _, _, _, v_targ = actor_critic(x2_ph, a_ph, z_ph, **ac_kwargs)

    # Experience buffer
    RB = get_replay_buffer(rb_type)
    replay_buffer = RB(obs_dim, act_dim, **rb_kwargs)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main/v', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t v: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q and V function
    q_backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*v_targ)
    min_q_pi = tf.minimum(q1_pi, q2_pi)
    v_backup = tf.stop_gradient(min_q_pi)

    # TD3 losses
    pi_loss = -tf.reduce_mean(q1_pi)
    q1_loss = 0.5 * tf.reduce_mean((q1 - q_backup)**2)
    q2_loss = 0.5 * tf.reduce_mean((q2 - q_backup)**2)
    v_loss = 0.5 * tf.reduce_mean((v - v_backup)**2)
    value_loss = q1_loss + q2_loss + v_loss

    # Separate train ops for pi, q
    policy_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    value_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_policy_op = policy_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    if ac_kwargs["pi_separate"]:
        train_policy_emb_op = policy_optimizer.minimize(pi_loss, var_list=get_vars('main/pi/emb'))
        train_policy_d_op = policy_optimizer.minimize(pi_loss, var_list=get_vars('main/pi/d'))
    train_value_op = value_optimizer.minimize(value_loss, var_list=get_vars('main/q') + get_vars('main/v'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main) 
                            for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def sample_z(size):
        if z_type == "uniform":
            return np.random.random_sample(size=size)
        elif z_type == "gaussian":
            return np.random.normal(size=size)
        else:
            raise Exception("z_type error")

    def get_action(o, noise_scale):
        pi_a = sess.run(pi, feed_dict={
            x_ph: o.reshape(1, -1), 
            z_ph: sample_z((1, z_dim))
        })[0]
        pi_a += noise_scale * np.random.randn(act_dim)
        pi_a = np.clip(pi_a, 0, 1)
        real_a = pi_a * act_high
        return pi_a, real_a

    def test_agent(n=10):
        test_actions = []
        for j in range(n):
            test_actions_ep = []
            o, r, d, ep_ret, ep_len = test_env.reset(), 0, False, 0, 0
            while not(d or (ep_len == test_max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                if test_without_state:
                    _, real_a = get_action(np.zeros(o.shape), 0)
                else:
                    _, real_a = get_action(o, 0)
                test_actions_ep.append(real_a)
                o, r, d, _ = test_env.step(real_a)
                ep_ret += r
                ep_len += 1
            logger.store(TestEpRet=ep_ret, TestEpLen=ep_len)
            test_actions.append(test_actions_ep)
        return test_actions

    start_time = time.time()
    o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    total_steps = steps_per_epoch * epochs

    rewards = []
    rets = []
    test_rets = []
    max_ret = None
    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        """
        Until start_steps have elapsed, randomly sample actions
        from a uniform distribution for better exploration. Afterwards, 
        use the learned policy (with some noise, via act_noise). 
        """
        if t > start_steps:
            pi_a, real_a = get_action(o, act_noise)
        else:
            pi_a, real_a = env.action_space.sample()

        # Step the env
        o2, r, d, _ = env.step(real_a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, pi_a, r, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        if d or (ep_len == max_ep_len):

            for _ in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {
                    x_ph: batch['obs1'],
                    x2_ph: batch['obs2'],
                    a_ph: batch['acts'],
                    r_ph: batch['rews'],
                    d_ph: batch['done']
                }
                feed_dict[z_ph] = sample_z((batch_size, z_dim))

                # Policy Learning update
                for key in feed_dict:
                    feed_dict[key] = np.repeat(feed_dict[key], q_pi_sample_size, axis=0)
                feed_dict[z_ph] = sample_z((batch_size*q_pi_sample_size, z_dim))
                if ac_kwargs["pi_separate"]:
                    if len(rewards) % 2 == 0:
                        outs = sess.run([pi_loss, train_policy_emb_op], feed_dict)
                    else:
                        outs = sess.run([pi_loss, train_policy_d_op], feed_dict)
                else:
                    outs = sess.run([pi_loss, train_policy_op], feed_dict)
                logger.store(LossPi=outs[0])

                # Q-learning update
                outs = sess.run([q1_loss, v_loss, q1, v, train_value_op], feed_dict)
                logger.store(LossQ=outs[0], LossV=outs[1], ValueQ=outs[2], ValueV=outs[3])


            logger.store(EpRet=ep_ret, EpLen=ep_len)
            rewards.append(ep_ret)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_actions = test_agent(number_of_tests_per_epoch)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            ret = logger.log_tabular('EpRet', average_only=True)[0]
            test_ret = logger.log_tabular('TestEpRet', average_only=True)[0]
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('LossV', average_only=True)
            logger.log_tabular('ValueQ', average_only=True)
            logger.log_tabular('ValueV', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            rets.append(ret)
            test_rets.append(test_ret)

            if max_ret is None or test_ret > max_ret:
                max_ret = test_ret
                best_test_actions = test_actions

            max_ep_len += inc_ep
            sess.run(target_update, feed_dict)

    logger.save_state({
        "rewards": rewards,
        "best_test_actions": best_test_actions,
        "rets": rets,
        "test_rets": test_rets,
        "max_ret": max_ret
    }, None)
    
    util.plot_actions(best_test_actions, act_high, logger.output_dir + '/best_test_actions.png')
    logger.log("max ret: %f" % max_ret)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('exp_name', type=str)
    parser.add_argument('model_config', type=str)
    parser.add_argument('env_config', type=str)
    parser.add_argument('seed', type=int, default=1007)
    args = parser.parse_args()

    import json
    model_config = json.load(open(util.MODEL_CONFIG_DIR + "iac/" + args.model_config))

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=util.LOG_DIR+os.path.splitext(args.env_config)[0])

    iac(
        env_config=util.ENV_CONFIG_DIR+args.env_config, 
        ac_type=model_config["ac_type"],
        ac_kwargs=model_config["ac_kwargs"],
        rb_type=model_config["rb_type"],
        rb_kwargs=model_config["rb_kwargs"],
        gamma=model_config["gamma"],
        lr=model_config["lr"],
        polyak=model_config["polyak"],
        batch_size=model_config["batch_size"],
        epochs=model_config["epochs"],
        start_steps=model_config["start_steps"],
        steps_per_epoch=model_config["steps_per_epoch"],
        max_ep_len=model_config["max_ep_len"],
        inc_ep=model_config["inc_ep"],
        q_pi_sample_size=model_config["q_pi_sample_size"],
        z_dim=model_config["z_dim"],
        z_type=model_config["z_type"],
        act_noise=model_config["act_noise"],
        number_of_tests_per_epoch=model_config["number_of_tests_per_epoch"],
        test_max_ep_len=model_config["test_max_ep_len"],
        test_without_state=model_config["test_without_state"],
        logger_kwargs=logger_kwargs,
        seed=args.seed
    )