import os
import numpy as np
import tensorflow as tf
import time

import core
from core import get_vars
from logx import EpochLogger
from env import make_env
from replay_buffer import get_replay_buffer
import util


def ddpg(
        env_config, ac_type, ac_kwargs, rb_type, rb_kwargs, 
        gamma, lr, polyak, batch_size, 
        epochs, start_steps, steps_per_epoch, inc_ep, 
        max_ep_len, test_max_ep_len, number_of_tests_per_epoch, act_noise,
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

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_high = env.action_space.high

    # Inputs to computation graph
    x_ph, a_ph, x2_ph, r_ph, d_ph = core.placeholders(obs_dim, act_dim, obs_dim, None, None)

    actor_critic = core.get_ddpg_actor_critic(ac_type)
    # Main outputs from computation graph
    with tf.variable_scope('main'):
        pi, q, q_pi = actor_critic(x_ph, a_ph, **ac_kwargs)
    
    # Target networks
    with tf.variable_scope('target'):
        pi_targ, _, q_pi_targ  = actor_critic(x2_ph, a_ph, **ac_kwargs)

    # Experience buffer
    RB = get_replay_buffer(rb_type)
    replay_buffer = RB(obs_dim, act_dim, **rb_kwargs)

    # Count variables
    var_counts = tuple(core.count_vars(scope) for scope in ['main/pi', 'main/q', 'main'])
    print('\nNumber of parameters: \t pi: %d, \t q: %d, \t total: %d\n'%var_counts)

    # Bellman backup for Q function
    backup = tf.stop_gradient(r_ph + gamma*(1-d_ph)*q_pi_targ)

    # DDPG losses
    pi_loss = -tf.reduce_mean(q_pi)
    q_loss = tf.reduce_mean((q-backup)**2)

    # Separate train ops for pi, q
    pi_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    q_optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    train_pi_op = pi_optimizer.minimize(pi_loss, var_list=get_vars('main/pi'))
    train_q_op = q_optimizer.minimize(q_loss, var_list=get_vars('main/q'))

    # Polyak averaging for target variables
    target_update = tf.group([tf.assign(v_targ, polyak*v_targ + (1-polyak)*v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    # Initializing targets to match main variables
    target_init = tf.group([tf.assign(v_targ, v_main)
                              for v_main, v_targ in zip(get_vars('main'), get_vars('target'))])

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(target_init)

    def get_action(o, noise_scale):
        pi_a = sess.run(pi, feed_dict={x_ph: o.reshape(1, -1)})[0]
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

    actions = []
    epoch_actions = []
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
        epoch_actions.append(pi_a)

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
            """
            Perform all DDPG updates at the end of the trajectory,
            in accordance with tuning done by TD3 paper authors.
            """
            for _ in range(ep_len):
                batch = replay_buffer.sample_batch(batch_size)
                feed_dict = {
                    x_ph: batch['obs1'],
                    x2_ph: batch['obs2'],
                    a_ph: batch['acts'],
                    r_ph: batch['rews'],
                    d_ph: batch['done']
                }

                # Q-learning update
                outs = sess.run([q_loss, q, train_q_op], feed_dict)
                logger.store(LossQ=outs[0], QVals=outs[1])

                # Policy update
                outs = sess.run([pi_loss, train_pi_op, target_update], feed_dict)
                logger.store(LossPi=outs[0])

            logger.store(EpRet=ep_ret, EpLen=ep_len)
            actions.append(np.mean(epoch_actions))
            epoch_actions = []
            rewards.append(ep_ret)
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        # End of epoch wrap-up
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent.
            test_actions = test_agent(number_of_tests_per_epoch)

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            ret = logger.log_tabular('EpRet', average_only=True)
            test_ret = logger.log_tabular('TestEpRet', average_only=True)[0]
            logger.log_tabular('EpLen', average_only=True)
            logger.log_tabular('TestEpLen', average_only=True)
            logger.log_tabular('QVals', average_only=True)
            logger.log_tabular('LossPi', average_only=True)
            logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()

            rets.append(ret)
            test_rets.append(test_ret)

            if max_ret is None or test_ret > max_ret:
                max_ret = test_ret
                best_test_actions = test_actions

            max_ep_len += inc_ep
            util.plot_actions(test_actions, act_high, logger.output_dir + '/actions%s.png' % epoch)

    logger.save_state({
        "actions": actions,
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
    model_config = json.load(open(util.MODEL_CONFIG_DIR + "ddpg/" + args.model_config))

    from logx import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed, data_dir=util.LOG_DIR+os.path.splitext(args.env_config)[0])

    ddpg(
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
        act_noise=model_config["act_noise"],
        number_of_tests_per_epoch=model_config["number_of_tests_per_epoch"],
        test_max_ep_len=model_config["test_max_ep_len"],
        logger_kwargs=logger_kwargs,
        seed=args.seed
    )