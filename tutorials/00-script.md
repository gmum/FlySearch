# `drone.py` parameters

## Scenario-related parameters 

* `scenario_type` -- either `forest_random`, `city_random`, `forest_random_anomaly`, `city_random_anomaly` or `mimic`. The first four mean that scenarios will be randomly generated, while the last means that scenario configurations will be copied (mimicked) from a known, existing run. Typically, you would use `mimic` and specify `mimic_run_path` to point to one of template directories in `run_templates` directory.
* `mimic_run_path` -- use only if `scenario_type` is set to `mimic`. This is the path to the directory where the run to be mimicked is stored.
* `mimic_run_cls_names` -- for normal use should set it to `*`. You only need to change it if for some reason you would like to perform a mimic run, but only for a certain class of objects (for example, fire). The argument is a list of semicolon (`;`) separated class names of objects that a scenario should contain in order to be mimicked (e.g. `fire` or `fire;car`. The name needs to be a _substring_ of the class name, so `car` would match all `car`-related classes (like sports cars, police cars and so on)). 
* `line_of_sight_assured` -- whether the agent should be able to see the searched object at the start of the trajectory. Should be set to `true` while generating FS-1-like and FS-A-1-like scenarios, and `false` for FS-2-like scenarios.

## Method-related parameters 

* `model` -- the model to be used. See  `models` subsection for more details.
* `agent` -- for normal use (i.e. evaluating VLMs) should be set to `simple_llm`. Refer to `tutorials/agents.md` for more details.
* `prompt_type` -- either `fs1` or `fs2`. FS-Anomaly-1 should use the same prompt type as FS-1.

## Logging-related parameters 

* `log_directory` -- name of the directory (relative to this script) where you wish to keep logs from runs. We recommend to keep it called `all_logs`.
* `run_name` -- all trajectories will be saved in `log_directory/run_name`. 

## Benchmark-related parameters

* `dummy_first` -- for normal use, should be set to `true`. Controls whether we discard first trajectory for the simulation environment to "warm up" (in case it misses assets and so on). Note that mimic runs assume this behaviour and duplicate the first trajectory config to compensate for that.
* `forgiveness` -- for normal use, should be set to `5`. This is amount of consecutive validation errors of model's actions. Exceeding that number will result in trajectory termination. 
* `glimpses` -- number of glimpses (or images) the agent is allowed to see in the trajectory. In our experiments, we used 10 glimpses for FS-1 and FS-Anomaly-1 and 20 glimpses for FS-2. Note that image of the class being searched in FS-2 is not counted towards this limit (even though that image is a first image logged in the trajectory).
* `number_of_runs` -- number of trajectories you wish to have generated (or replicated from the mimic run).
* `show_class_image` -- whether the agent should receive additional, visual prompt containing image of the object being searched. Should be set to `true` while generating FS-2-like scenarios, and `false` for FS-1-like and FS-A-1-like scenarios.
* `continue_from` -- useful if your run was interrupted for some reason, and you want to continue it. For example, if your run was interrupted during 42nd trajectory, you would need to delete its folder from logs (as it is unfinished and needs to be redone) and set `continue_from` to `42`. 