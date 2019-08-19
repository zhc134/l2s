import numpy as np
from matplotlib import pyplot as plt
plt.switch_backend('agg')

DATA_DIR = "data/"
CONFIG_DIR = "config/"
ENV_CONFIG_DIR = CONFIG_DIR + "envs/"
MODEL_CONFIG_DIR = CONFIG_DIR + "models/"
GS_CONFIG_DIR = CONFIG_DIR + "gs/"
SETTING_DIR = DATA_DIR + "setting/"
LOG_DIR= DATA_DIR + "log/"
REPORT_DIR = DATA_DIR + "report/"

def plot_actions(actions, act_high, fig_file):
    actions = np.array(actions)
    if len(actions.shape) == 2:
        _, n_params = actions.shape
        plt.figure(figsize=(n_params*8, 5))
        for j in range(n_params):
            plt.subplot(1, n_params, j+1)
            plt.hist(actions[:, j].flatten(), bins=100, range=(0, act_high[j]))
            plt.title("param %d" % j)
    else:
        n_test, _, n_params = actions.shape
        plt.figure(figsize=(n_params*8, (n_test+1)*5))
        for j in range(n_params):
            plt.subplot(n_test+1, n_params, j+1)
            plt.hist(actions[:, :, j].flatten(), bins=100, range=(0, act_high[j]))
            plt.title("total of param %d" % j)
        for i in range(n_test):
            for j in range(n_params):
                plt.subplot(n_test+1, n_params, (i+1)*n_params+j+1)
                plt.hist(actions[i, :, j], bins=100, range=(0, act_high[j]))
                plt.title("test %d param %d" % (i, j))
    plt.savefig(fig_file)
    plt.close()

def plot_adv(act, adv, fig_file):
    _, n_params = act.shape
    m = 4
    idx = abs(adv - np.mean(adv)) < m * np.std(adv)
    act = act[idx]
    adv = adv[idx]

    plt.figure(figsize=(n_params*8, 5))
    for i in range(n_params):
        nbins = 20
        act_i = act[:, i]
        n, _ = np.histogram(act_i, bins=nbins)
        sy, _ = np.histogram(act_i, bins=nbins, weights=adv)
        sy2, _ = np.histogram(act_i, bins=nbins, weights=adv*adv)
        mean = sy / n
        std = np.sqrt(sy2/n - mean*mean)

        plt.subplot(1, n_params, i+1)
        plt.plot(act_i, adv, 'bo', zorder=-1)
        plt.errorbar((_[1:] + _[:-1])/2, mean, yerr=std, fmt='r-', capsize=4)
        plt.title("param %d" % i)
    plt.savefig(fig_file)
    plt.close()

def plot_seq_actions(actions, act_high, fig_file):
    actions = np.array(actions)
    _, n_params = actions.shape
    plt.figure(figsize=(n_params*8, 5))
    for j in range(n_params):
        plt.subplot(1, n_params, j+1)
        plt.plot(actions[:, j])
        plt.title("param %d" % j)
    plt.savefig(fig_file)
    plt.close()

def plot_seq_obs_and_actions(obs, actions, act_high, fig_file):
    obs = np.array(obs)
    actions = np.array(actions)
    _, n_params = actions.shape
    plt.figure(figsize=(n_params*8, 5))
    for j in range(n_params):
        ax1 = plt.subplot(1, n_params, j+1)
        ax1.plot(actions[:, j], 'b-')

        ax2 = ax1.twinx()
        ax2.plot(obs[:, 0], 'r-')
        plt.title("param %d" % j)
    plt.savefig(fig_file)
    plt.close("all")
