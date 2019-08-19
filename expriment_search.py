from multiprocessing.pool import ThreadPool
import subprocess
import argparse
import json
import os
import numpy as np
import pandas as pd
import traceback
import math
import itertools
json.encoder.FLOAT_REPR = lambda x: format(x, '.1f')

true_stds = [0.0]
fit_stds = [0.0, 0.5, 1.0]
probs = [[0.5, 0.5], [0.3, 0.7]]

truth_search_params = [
    {"maxSpeed": [10., 30., 6.0, 2.0], "maxNegAcc": [3.0, 6.0, 1.5, 0.2]},
    # {"maxPosAcc"  : [1.5, 4.5, 1.0, 0.3]},
    # {"maxNegAcc"  : [3.0, 6.0, 1.0, 0.3]},
    # {"usualNegAcc": [1.5, 4.5, 1.0, 0.3]},
    # {"minGap"     : [1.5, 5.0, 1.0, 0.2]},
    # {"maxSpeed"   : [10., 30., 4.0, 1.0]},
]

action_space = {
    "maxPosAcc"  : 6.0,
    "maxNegAcc"  : 8.0, 
    "usualNegAcc": 6.0,
    "minGap"     : 6.0,
    "maxSpeed"   : 40.0,
}

def checkCall(command, dir, workdir="."):
    _command = command.split()
    print('[Start]  '+ command)
    try:
        subprocess.check_call(_command, stdout=open(os.path.join(dir, "log.txt"), "w"), stderr=subprocess.STDOUT, cwd=workdir)
        print('[Finish] '+ command)
    except subprocess.CalledProcessError:
        print('[Error]  '+ command)

def search(thread_num):
    commands = []
    with open("config/envs/base.json", "r") as f_env:
        base_config = json.load(f_env)
    for ground_truth in truth_search_params:
        env_config = base_config
        env_config["target_params"] = []
        env_config["action_space"] = {}
        env_config["action_space"]["high"] = []
        envs = []
        configs = {}
        
        for env, params in ground_truth.items():
            configs[env] = []
            env_config = base_config
            env_config["target_params"].append(env)
            env_config["action_space"]["high"].append(action_space[env])
            envs.append(env)
            for low in np.arange(params[0], params[1], params[3]):
                for high in np.arange(low + params[2], params[1], params[3]):
                    for _true_std in true_stds:
                        true_std = _true_std * np.mean([params[0], params[1]])
                        config = {
                            "target_params" : env,
                            "stds"          : true_std,
                            "means"         : [low, high],
                        }
                        configs[env].append(config)
            with open("config/envs/" + '_'.join(envs) + ".json", "w") as f_env:
                json.dump(env_config, f_env, sort_keys=True, indent=4, separators=(',', ': '))
        
        def iter(configs):
            keys, values = zip(*configs.items())
            for v in itertools.product(*values):
                yield dict(zip(keys, v))
        for config in iter(configs):
            for prob in probs:
                for fit_std in fit_stds:
                    # print(config)
                    path = ""
                    config_file = {"means" : [[], []], "probs": prob, "stds": [], "target_params": [], "fit_std": fit_stds}
                    
                    for env, params in config.items():
                        low, high = params["means"]
                        config_file["means"][0].append(low)
                        config_file["means"][1].append(high)
                        config_file["target_params"].append(env)
                        true_std = params["stds"]
                        config_file["stds"].append(params["stds"])
                    
                        path = path + "%s_%.1f_%.1f_%.1f_%.2f_" % (env, low, high, prob[0], true_std)

                    path = path + "%.1f" % (fit_std)
                    path = os.path.join("logs", "experiment_setting", path)
                    if not os.path.exists(path):
                        os.makedirs(path)
                    command = "python3 grid_search_agent.py --env %s --agent --std %.2f --test --log --dir %s" % (env, fit_std, path)
                    with open(os.path.join(path, "config.json"), "w") as f_config:
                        json.dump(config_file, f_config, sort_keys=True, indent=4, separators=(',', ': '))
                    with open(os.path.join(path, "command.txt"), "w") as f_command:
                        f_command.write(command)
                    commands.append([command, path])

    print("Parameters are ready! Total settings: %d" % len(commands))
    tp = ThreadPool(args.thread_num)
    for command, path in commands:
        tp.apply_async(checkCall, (command, path, ))
    
    tp.close()
    tp.join()

def analysis(save_report):
    log_dir = "logs/experiment_setting"
    data = []
    for exp in next(os.walk(log_dir))[1]:
        try:
            config = json.load(open("%s/%s/config.json" % (log_dir, exp)))
            with open("%s/%s/final.txt" % (log_dir, exp)) as f:
                ret, test_r, grid_r = map(float, f.readline().split())
            if math.isnan(ret):
                ret = -1000
            row = [ret, np.around(test_r, decimals=5), np.around(grid_r, decimals=5)]
            row.append(str(config["target_params"]))
            row.append(np.around(config["means"], decimals=1))
            row.append(np.around(config["probs"], decimals=1))
            row.append(np.around(config["stds"], decimals=1))
            row.append(exp)
            data.append(row)
        except Exception as e:
            print(e)

    columns = ["result", "test_r", "grid_r", "target_params", "means", "probs", "true_stds", "path"]
    df = pd.DataFrame(data, columns=columns)
    df = df.astype(str)
    df["result"] = pd.to_numeric(df["result"])
    df.sort_values("result", ascending=False, inplace=True)

    print(df.head())
    REPORT_DIR = "logs/experiment_setting/"
    if save_report:
        report_file = REPORT_DIR + "report.txt"
        df.to_string(open(report_file, "w"))
        print("report saved to " + report_file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--thread_num', type=int, default=1)
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--save_report', action='store_true')
    args = parser.parse_args()
    if args.search:
        if args.thread_num > 0:
            search(args.thread_num)
        else:
            print("Plase given a valid thread number by --thread_num")
    analysis(args.save_report)
