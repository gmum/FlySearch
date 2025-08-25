# FlySearch

This is repository contains code for the FlySearch benchmark ([arXiv link](https://arxiv.org/abs/2506.02896v2)).

## Abstract 

The real world is messy and unstructured. Uncovering critical information often requires active, goal-driven exploration. It remains to be seen whether Vision-Language Models (VLMs), which recently emerged as a popular zero-shot tool in many difficult tasks, can operate effectively in such conditions. In this paper, we answer this question by introducing FlySearch, a 3D, outdoor, photorealistic environment for searching and navigating to objects in complex scenes. We define three sets of scenarios with varying difficulty and observe that state-of-the-art VLMs cannot reliably solve even the simplest exploration tasks, with the gap to human performance increasing as the tasks get harder. We identify a set of central causes, ranging from vision hallucination, through context misunderstanding, to task planning failures, and we show that some of them can be addressed by finetuning. We publicly release the benchmark, scenarios, and the underlying codebase.


## Dependencies

We've verified that the benchmark works on Ubuntu 22.04. 

### Python environment 

We suggest you use Python 3.12 and then install dependencies using `pip install -r requirements.txt`.

### `.env` file 

Before proceeding, you need to create a `.env` file in the root directory of this repository. We've provided a template for it in the file `.env-example`. In other words, you should run:

```bash 
cp .env-example .env
```

You will need to edit the `.env` file so that it contains your local variables (such as paths to binaries, fonts, etc.). We will describe the variables you need to set below.

### Benchmark binaries

Binaries cam be downloaded from https://doi.org/10.5281/zenodo.15428224.

`city.tar.gz` contains the city environment and
`forest.tar.gz` contains the forest environment. Extract them and then modify the `.env` file by:

* setting the `CITY_BINARY_PATH` to `/your_location/simulator/CitySample/Binaries/Linux/CitySample`
* setting the FOREST_BINARY_PATH to `your_location/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample`

You can also verify manually that these work on your computer by
running `./simulator/CitySample/Binaries/Linux/CitySample` or
`./simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample`. These commands should start the UE5
environment and show it on your screen.

### City locations

The file `locations_city.csv` is provided with this repository. Set the `LOCATIONS_CITY_PATH` variable in the `.env` file to location of this file in your filesystem; this file is important for running city scenarios, as it contains permissible safe locations for objects to be spawned.

### Fonts

The benchmark needs a font to overlay images from the engine with a navigation scaffold. Set the `FONT_LOCATION` variable in the `.env` file to the location of a font file in your filesystem. The default one is `/usr/share/fonts/google-noto/NotoSerif-Bold.ttf`, which may or may not be present on your machine.

### API keys

To use closed-source VLMs, you need to have an API key. To configure them, set appropriate variables in the
`.env` file.


## Running FlySearch

### Examples 

We present examples on how to run FlySearch benchmark in the `examples` directory.

- `mimic.sh` contains 3 examples of how to run different models on FS-1, FS-Anomaly-1 and FS-2.
- `create_new.sh` contains 3 examples of how to sample new scenarios from distribution of scenarios used in FS-1, FS-Anomaly-1 and FS-2. 
If you just want to test a model on our benchmark, we recommend using `mimic.sh`.


### Direct `drone.py` script usage

`drone.py` script is the entrypoint to run the benchmark. It has several configurable parameters. 

#### Scenario-related parameters 

* `scenario_type` -- either `forest_random`, `city_random` or `mimic`. The first two mean that scenarios will be randomly generated, while the last means that scenario configurations will be copied (mimicked) from a known, existing run. Typically, you would use `mimic` and specify `mimic_run_path` to point to one of template directories in `run_templates` directory.
* `mimic_run_path` -- use only if `scenario_type` is set to `mimic`. This is the path to the directory where the run to be mimicked is stored.
* `mimic_run_cls_names` -- for normal use should set it to `*`. You only need to change it if for some reason you would like to perform a mimic run, but only for a certain class of objects (for example, fire). The argument is a list of semicolon (`;`) separated class names of objects that a scenario should contain in order to be mimicked (e.g. `fire` or `fire;car`. The name needs to be a _substring_ of the class name, so `car` would match all `car`-related classes (like sports cars, police cars and so on)). 
* `line_of_sight_assured` -- whether the agent should be able to see the searched object at the start of the trajectory. Should be set to `true` while generating FS-1-like and FS-A-1-like scenarios, and `false` for FS-2-like scenarios.

#### Method-related parameters 

* `model` -- the model to be used. See  `models` subsection for more details.
* `agent` -- for normal use (i.e. evaluating VLMs) should be set to `simple_llm`. Refer to `tutorials/agents.md` for more details.
* `prompt_type` -- either `fs1` or `fs2`.

#### Logging-related parameters 

* `log_directory` -- name of the directory (relative to this script) where you wish to keep logs from runs. We recommend to keep it called `all_logs`.
* `run_name` -- all trajectories will be saved in `log_directory/run_name`. 

#### Benchmark-related parameters

* `dummy_first` -- for normal use, should be set to `true`. Controls whether we discard first trajectory for the simulation environment to "warm up" (in case it misses assets and so on). Note that mimic runs assume this behaviour and duplicate the first trajectory config to compensate for that.
* `forgiveness` -- for normal use, should be set to `5`. This is amount of consecutive validation errors of model's actions. Exceeding that number will result in trajectory termination. 
* `glimpses` -- number of glimpses (or images) the agent is allowed to see in the trajectory. In our experiments, we used 10 glimpses for FS-1 and FS-Anomaly-1 and 20 glimpses for FS-2. Note that image of the class being searched in FS-2 is not counted towards this limit (even though that image is a first image logged in the trajectory).
* `number_of_runs` -- number of trajectories you wish to have generated (or replicated from the mimic run).
* `show_class_image` -- whether the agent should receive additional, visual prompt containing image of the object being searched. Should be set to `true` while generating FS-2-like scenarios, and `false` for FS-1-like and FS-A-1-like scenarios.
* `continue_from` -- useful if your run was interrupted for some reason, and you want to continue it. For example, if your run was interrupted during 42nd trajectory, you would need to delete its folder from logs (as it is unfinished and needs to be redone) and set `continue_from` to `42`. 

#### Models 

FlySearch supports testing several models: 
* GPT-4o. To use it, set `model` flag to `gpt-4o`.
* Gemini family models (e.g gemini-2.0-flash. To use it, set `model` flag to `gemini-2.0-flash`.)
* Anthropic models. To use it, prefix the model name with `anthropic-`, e.g. `anthropic-claude-3-5-sonnet-20241022`.
* Any models behind a VLLM API. If model name does not match any of the above, it is assumed to be a VLLM model. OpenAI protocol is used to communicate with the model. For example, to use Gemma3-27b hosted on DeepInfra ([link](https://deepinfra.com/)), you need to configure `.env` file with `VLLM_ADDRESS = 'https://api.deepinfra.com/v1/openai'`, `VLLM_KEY` matching your DeepInfra API key, and set `model` to `google/gemma-3-27b-it`.

## Direct use of our environment (without code for benchmarking)

Our environment is based on Unreal Engine 5, and we provide a Gymnasium-like interface to it. It can be used directly without the rest of the code in this repository. See `tutorials/01-environment.md` for examples. 

## Notes

### UE5 binary crashes 

The UE5 binary can sometimes spontaneously crash. The code is designed to handle this (we've modified UnrealCV's code to do so), but in case it happens you just need to restart the script with appropriately set `continue_from` flag.