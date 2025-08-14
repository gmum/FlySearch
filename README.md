# FlySearch

This is repository contains code for the FlySearch benchmark ([arXiv link](https://arxiv.org/abs/2506.02896v2)).

## Dependencies

To be able to run the benchmark, you need to download appropriate dependencies and configure some variables. The purpose
of this section is to tell you how to do that.

### Python environment 

We suggest you use Python 3.12 and then install dependencies using `pip install -r requirements.txt`.

### Benchmark binaries

Can be downloaded from https://doi.org/10.5281/zenodo.15428224.

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


## Entrypoint

We recommend using the `drone.sh` script to run the benchmark. It is a Bash script that calls the `drone.py` script
which performs the entire evaluation. It has several configurable parameters, such as:

* `scenario_type` -- either `forest_random`, `city_random` or `mimic`. The first two mean that scenarios will be randomly generated, while the last means that scenario configurations will be copied (mimicked) from a known, existing run.
* `mimic_run_path` -- use only if `scenario_type` is set to `mimic`. This is the path to the directory where the run to be mimicked is stored.
* `mimic_run_cls_names` -- for normal use should be set to `*` (as is)
* `model` -- the model to be used. GPT-4o will call OpenAI's API, while prefixing model name with `anthropic-` will assume that Anthropic's library needs to be used. If `gemini` is present in models name, we assume it's an appropriate Gemini model. To use Sonnet, use `anthropic-claude-3-5-sonnet-20241022`. If model is not recognized, the script assumes that it's a VLLM model.
* `log_directory` -- name of the directory (relative to this script) where you wish to keep logs from runs. We recommend to keep it called `all_logs`
* `run_name` -- all trajectories will be saved in `log_directory/run_name`.
* `dummy_first` -- for normal use, should be set to `true`. Controls whether we discard first trajectory for the simulation environment to "warm up" (in case it misses assets and so on). Note that mimic runs assume this behaviour and duplicate the first trajectory config to compensate for that.
* `forgiveness` -- for normal use, should be set to `5`. This is amount of consecutive validation errors of model's actions.
* `glimpses` -- number of glimpses (or images) the agent is allowed to see in the trajectory. In our experiments, we used 10 glimpses for FS-1 and FS-Anomaly-1 and 20 glimpses for FS-2.
* `number_of_runs` -- number of trajectories you wish to have generated.
* `agent` -- for normal use should be set to `simple_llm`.
* `line_of_sight_assured` -- whether the agent should be able to see the searched object at the start of the trajectory. Should be set to `true` while generating FS-1-like and FS-A-1-like scenarios, and `false` for FS-2-like scenarios.
* `show_class_image` -- whether the agent should receive additional, visual prompt containing image of the object being searched. Should be set to `true` while generating FS-2-like scenarios, and `false` for FS-1-like and FS-A-1-like scenarios.
* `prompt_type` -- either `fs1` or `fs2`.
