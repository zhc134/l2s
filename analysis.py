import os.path as osp, os
import json
import pandas as pd
import joblib
import util

DEFAULT_GS_CONFIG = "sddpg/default.json"

def anaylsis(env_name, exp_name, gs_config=None, save_report=False):
    print('Generating Report...')
    groupby = False
    if gs_config:
        groupby = True
    else:
        gs_config = DEFAULT_GS_CONFIG

    config = json.load(open(util.GS_CONFIG_DIR + gs_config))
    params = list(config["grid"].keys())

    data_dir = osp.join(util.LOG_DIR, env_name, exp_name)
    data = []

    def get_param(config, key):
        for k in key.split('.'):
            config = config[k]
        return config

    for exp in next(os.walk(data_dir))[1]:
        try:
            var = joblib.load("%s/%s/vars.pkl" % (data_dir, exp))
            config = json.load(open("%s/%s/config.json" % (data_dir, exp)))
            row = [var["max_ret"], exp]
            for param in params:
                for key in param.split(','):
                    row.append(get_param(config, key))
            data.append(row)
        except Exception as e:
            print(e)

    columns = []
    if params:
        for param in params:
            for key in param.split(','):
                columns.append(key)
    
    df = pd.DataFrame(data, columns=["max_ret", "name"] + columns)
    df = df.astype(str)
    df["max_ret"] = pd.to_numeric(df["max_ret"])
    if groupby:
        res = df.groupby(columns)["max_ret"].agg(["mean", "std", "max", "min"])
        res.columns = ["max_ret", "std", "max", "min"]
        res = res.reset_index()
        columns = list(res.columns)
        columns = columns[-4:] + columns[:-4]
        res = res[columns]
    else:
        res = df
    res.sort_values("max_ret", ascending=False, inplace=True)

    print(res.head())
    if save_report:
        if not osp.exists(util.REPORT_DIR):
            os.makedirs(util.REPORT_DIR)
        report_file = util.REPORT_DIR + env_name + "_" + exp_name + ".txt"
        res.to_string(open(report_file, "w"))
        print("report saved to " + report_file)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('exp_name', type=str)
    parser.add_argument('--gs_config', type=str, default=None)
    parser.add_argument('--save_report', action="store_true")
    args = parser.parse_args()
    
    if args.gs_config is None:
        input(
            "No Grid Search Config Provided. Using sddpg/default.json.\n" 
            "Groupby params disabled. Confirm? or ctrl+C to abort.")
    anaylsis(args.env_name, args.exp_name, args.gs_config, args.save_report)