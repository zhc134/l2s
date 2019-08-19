import json
from git import Repo
import datetime
import subprocess
import os
import hashlib
import shutil
import sys

configName = ".config.json"
timeStamp = datetime.datetime.now()
experimentFilePath = "sddpg.py"
generatorPath = "generate_setting.py"
agenePath = "fix_sim_agent.py"
dataPath = "../data/setting/1x1"
dataFiles = ["flow_1_1.jsonl", "observed_1_1.txt", "roadnet_1_1.json", "signal_1_1.jsonl"]
pythonPath = "python3"


def checkCall(command, workdir="."):
    command = command.split()
    try:
        subprocess.check_call(command, cwd=workdir)
    except subprocess.CalledProcessError as e:
        print("executing error: " + str(e.__doc__) + str(e))
        sys.exit(-1)

def hashData(path=dataPath, files=dataFiles):
    sha1 = hashlib.sha1()
    for datafile in files:
        fullPath = os.path.join(dataPath, datafile)
        with open(fullPath, 'rb') as f:
            sha1.update(f.read())
    return sha1.hexdigest()

def get(msg, default=None):
    msg = msg + "[{}]: ".format(default) if default != None else msg + ": "
    while (True):
        data = input(msg)
        if data != "":
            return data
        elif default != None:
            return default

def getDescription(filename):
    description = {}
    print("Input Description in format \"xxx:xxx\", end with an empty line")
    while (True):
        line = input()
        if (line.strip() == ""):
            break
        try:
            x, y = line.split(':', 1)
            description[x] = y
        except:
            print("Invalid format")
    with open(filename, 'w') as f:
        json.dump(description, f)

def loadConfig(filename=configName):
    try:
        with open(filename) as f_config:
            config = json.load(f_config)
    except:
        config = {}

    if (not 'name' in config):
        config['name'] = get("Please input your name")
    
    print("Hello, {}!".format(config['name']))
    saveConfig(config)
    return config

def commitFile(config, path="../"):
    git = Repo(path).git
    git.add(".")
    defaultMsg = "{} {}".format(config['name'], timeStamp.strftime("%Y-%m-%d %H:%M:%S"))
    commitMsg = get("Commit message", default=defaultMsg)
    git.commit("-m", commitMsg)
    return git.log("--pretty=format:%H", "-1")

def generateData(config, commitHash):
    if config.get('exsitData', False):
        while (True):
            generate=get("Generate new data? y/n", default='n').lower()
            if generate in {'n', 'y'}:
                break
    else:
        generate = 'y'
    
    if generate == 'y':
        if ("lastGeneratorArgs" in config):
            generatorArgs = get("Args of generator", default="last time")
            generatorArgs = config["lastGeneratorArgs"] if generatorArgs == "last time" else generatorArgs
        else:
            generatorArgs = get("Args of generator")
        config["lastGeneratorArgs"] = generatorArgs
        checkCall("{} {} {}".format(pythonPath, generatorPath, generatorArgs), workdir="../tools")
        checkCall("{} {}".format(pythonPath, agenePath))
        hashCode = hashData(path=dataPath, files=dataFiles)
        with open(os.path.join(dataPath, "data_hash"), 'w') as HashFile:
            HashFile.write(hashCode)
            
        dataLogPath = os.path.join("logs", "datas", hashCode)
        if not os.path.exists(dataLogPath):
            os.makedirs(dataLogPath)
        with open(os.path.join(dataLogPath, "commit_hash"), 'w') as commitHashFile:
            commitHashFile.write(commitHash)
            
        with open(os.path.join(dataLogPath, "generate_args"), 'w') as ArgsFile:
            ArgsFile.write(generatorArgs)
        
        config['exsitData'] = True
    saveConfig(config)    
    return config

def runExperiment(config):
    experimentName = get("Experiment name")
    path = os.path.join("logs", config['name'], "{} {}".format(timeStamp.strftime("%Y-%m-%d %H:%M:%S"), experimentName))
    os.makedirs(path)

    commitHash = commitFile(config)
    with open(os.path.join(path, "commit_hash"), 'w') as commitHashFile:
        commitHashFile.write(commitHash)

    config = generateData(config, commitHash)
    shutil.copy(os.path.join(dataPath, 'data_hash'), path)

    getDescription(os.path.join(path, "description.json"))
    if ("lastExperimentArgs" in config):
        ExperimentArgs = get("Args of running experiment", default="last time")
        ExperimentArgs = config["lastExperimentArgs"] if ExperimentArgs == "last time" else ExperimentArgs
    else:
        ExperimentArgs = get("Args of Experiment", default="")
    config["lastGeneratorArgs"] = ExperimentArgs
    with open(os.path.join(path, "experiment_args"), 'w') as ArgsFile:
            ArgsFile.write(ExperimentArgs)
    
    checkCall("{} {} {}".format(pythonPath, experimentFilePath, ExperimentArgs))
    saveConfig(config)
    return config

def saveConfig(config, filename=configName):
    with open(filename, 'w') as f_config:
        json.dump(config, f_config)

if __name__ == '__main__':
    config = loadConfig()
    config = runExperiment(config)
    saveConfig(config)



