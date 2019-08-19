from multiprocessing.pool import ThreadPool
import subprocess
from itertools import product
import argparse
import json
import os.path as osp, os
import util
import uuid
import atexit
import glob
import time

def get_random_id():
    return str(uuid.uuid4()).split('-')[0]

def clean(model_config_dir, tmp_config_prefix):
    for config in glob.glob(model_config_dir + "/" + tmp_config_prefix + "*"):
        os.remove(config)

def call(command):
    print(*command)
    subprocess.call(command)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('thread_num', type=int)
    parser.add_argument('--report', action="store_true")
    args = parser.parse_args()

    config = json.load(open(util.GS_CONFIG_DIR + args.config))

    alg = config["alg"]
    gs_id = get_random_id()
    expname = config["expname"] + "_" + time.strftime("%Y-%m-%d_%H-%M-%S") + "_" + gs_id
    env_config = config["env_config"]

    input(
        "Notice!!!!\n"
        "You're running {} with:\n"
        "config: {}\n"
        "expname: {}\n"
        "env_config: {}\n"
        "Confirm? or ctrl+c to abort.\n"
        .format(alg, args.config, expname, env_config)
    )

    grid = config["grid"]
    model_config_dir = util.MODEL_CONFIG_DIR + alg + "/"
    model_default_config = json.load(open(model_config_dir + config["model_default_config"]))
    
    tmp_config_prefix = "gs-" + gs_id + "-"
    atexit.register(clean, model_config_dir, tmp_config_prefix)

    def set_model_config(config, key, value):
        for k in key.split('.')[:-1]:
            config = config[k]
        config[key.split('.')[-1]] = value
                
    commands = []
    keys, values = zip(*grid.items())
    for v in product(*values):
        tmp_config_file = tmp_config_prefix + get_random_id() + ".json"
        model_config = model_default_config.copy()
        for key, value in zip(keys, v):
            if ',' in key:
                key = key.split(',')
                for i, k in enumerate(key):
                    set_model_config(model_config, k, value[i])
            else:
                set_model_config(model_config, key, value)
        json.dump(model_config, open(model_config_dir + tmp_config_file, "w"))
        command = ["python", alg + ".py", expname, tmp_config_file, env_config, str(config["seed"])]
        for _ in range(config["run_per_config"]):
            commands.append(command)

    tp = ThreadPool(args.thread_num)

    for command in commands:
        tp.apply_async(call, (command,))

    tp.close()
    tp.join()

    if args.report:
        from analysis import anaylsis
        anaylsis(os.path.splitext(env_config)[0], expname, args.config, True)