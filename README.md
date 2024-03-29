## Folder Structure
```
├─config
│  ├─envs (config for environment)
│  ├─gs (config for grid search)
│  │  └─sddpg (algorithm name)
│  └─models (config for algorithm)
│      └─sddpg (algorithm)
├─data
│  ├─log
│  │  ├─1x1 (environment)
│  │  │  ├─2019-01-17_sddpg_sz (exp_name)
│  │  │  │  ├─2019-01-17_03-18-26-sddpg_sz_s1007_1790dfbd
│  ├─report (report generated by analysis.py or grid_search.py)
│  └─setting (environment files)
│      └─1x1 (environment)
├─tools 
```

## Workflow
1. use tools/generate_setting.py to generate `roadnet.json`, `signal.jsonl`, `flow.jsonl`
2. use one of the sim_agent to generate `observed.txt`
3. create corresponding `config/envs/env.json`
4. create model config `config/models/<alg>/<yourname>xxx.json`
5. create grid search config `config/gs/<alg>/<yourname>xxx.json`
6. run `grid_search.py` with grid search config, use `--report` flag to generate report

> Notice! \
> you can implement different **replay buffer** in `replay_buffer.py` \
> you can implement different **network architecture** in `core.py` \
> any modification to algorithm file should be noticed on Wechat.

All configs file are self-explained, if you have any questions, please contact me!

Happy Exploring!
