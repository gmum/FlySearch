# FlySearch

This is repository containing code for the FlySearch benchmark.

## Entrypoint

We recommend using the `drone.sh` script to run the benchmark. It is a Bash script that calls the `drone.py` script
which performs the entire evaluation. It has several configurable parameters, such as:

* `glimpses` -- number of glimpses (or images) the agent is allowed to see in the trajectory. In our experiments, we
  used
  10 glimpses.
* `glimpse_generator`, `prompt`, `navigator`, `incontext` -- these should be left as they are, being remnants of our
  previous
  experiments.
* `model` -- the model to be used. GPT-4o will call OpenAI's API, while prefixing model name with `anthropic-` will
  assume
  that Anthropic's library needs to be used. To use Sonnet, use `anthropic-claude-3-5-sonnet-20241022`. If model is not
  recognized, the script assumes that it's a VLLM model.
* `run_name` -- Name of the directory where the logs will be stored.
* `scenario_type` -- forest, city, forest-anomaly or city-anomaly.
* `height_min` -- The minimum height of the drone that the model can be spawned at.
* `height_max` -- The maximum height of the drone that the model can be spawned at.
* `repeats` -- Number of times any given scenario should be repeated. We recommend using 1.
* `logdir` -- Directory where all `run_name` directories will be stored.
* `n` -- Number of scenarios to be generated.
* `mimic_run_name` -- If this option is not empty, the script will copy configurations from the `mimic_run_name` run for
  each trajectory. Note that this may override some configuration options (but not all of them).
* `mimic_run_cls_names` -- Used to specify specific class names to be run during mimic run. Useful in case you need a
  rerun of a specific class. The syntax is `class1,class2,class3`. We recommend setting it to `*` to run all classes.
* `continue_from` -- runs the benchmark from the specified trajectory number. Useful in case you want to continue a
  previously interrupted run. Works with mimicking.

## Dependencies

To be able to run the benchmark, you need to download appropriate dependencies and configure some variables. The purpose
of this section is to tell you how to do that.

### Benchmark binaries

Can be downloaded from https://zenodo.org/records/14775310.

`city.tar.gz` contains the city environment and
`forest3.tar.gz` contains the forest environment. Extract them and then modify the `drone.sh` script by:

* setting the `CITY_BINARY_PATH` to `/your_location/simulator/CitySample/Binaries/Linux/CitySample`
* setting the FOREST_BINARY_PATH to
  `your_location/simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample`

That's all you need to do to configure the binaries!

You can also verify manually that these work on your computer by
running `./simulator/CitySample/Binaries/Linux/CitySample` or
`./simulator-dreamsenv/Linux/ElectricDreamsEnv/Binaries/Linux/ElectricDreamsSample`. These commands should start the UE5
environment and show it on your screen.

### City locations

The file `locations_city.csv` is provided with this repository. Set the `LOCATIONS_CITY_PATH` variable in the `drone.sh`
to location of this file in your filesystem; this file is important for running city scenarios, as it contains
permissible safe locations for objects to be spawned.

### Fonts

The benchmark needs a font to overlay images from the engine with a navigation scaffold. Set the `FONT_LOCATION`
variable in the `drone.sh` script to the location of a font file in your filesystem. The default one is
`/usr/share/fonts/google-noto/NotoSerif-Bold.ttf`, which may or may not be present on your machine.

### API keys

To use closed-source VLMs, you need to have an API key. To configure them, set appropriate variables in the
`misc/config.py` file.


