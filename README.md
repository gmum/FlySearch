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

`drone.py` script is the entrypoint to run the benchmark. It has several configurable parameters. We explain them in the `tutorials/00-script.md` file.

### Models 

FlySearch supports testing several models. To test them, you can use on of our examples (discussed in the `Examples` section) and change the `model` parameter appropriately:
* OpenAI models. To use it, prefix your model argument with `oai-` and then use OpenAI's model name (e.g. `oai-gpt-4o`).
* Gemini family models (e.g. gemini-2.0-flash. To use it, set `model` flag to `gemini-2.0-flash`). Note that we use Gemini models using compatibility mode with OpenAI format -- as of now, this unfortunately tends to fail with Gemini 2.5 Pro. Pull requests fixing that issue are welcome.
* Anthropic models. To use it, prefix the model name with `anthropic-`, e.g. `anthropic-claude-3-5-sonnet-20241022`.
* Any models behind a VLLM API. If model name does not match any of the above, it is assumed to be a VLLM model. OpenAI protocol is used to communicate with the model. For example, to use Gemma3-27b hosted on [DeepInfra](https://deepinfra.com/), you need to configure `.env` file with `VLLM_ADDRESS = 'https://api.deepinfra.com/v1/openai'`, `VLLM_KEY` matching your DeepInfra API key, and set `model` to `google/gemma-3-27b-it`.

### Direct use of our environment (without code for benchmarking)

Our environment is based on Unreal Engine 5, and we provide a Gymnasium-like interface to it. It can be used directly without the rest of the code in this repository. See `tutorials/01-environment.md` for examples. 

## Notes

### UE5 binary crashes 

The UE5 binary can sometimes spontaneously crash. The code is designed to handle this (we've modified UnrealCV's code to do so), but in case it happens you just need to restart the script with appropriately set `continue_from` flag. Furthermore, in case where your code was terminated by `UnrealDiedException` please open an issue here with a stack trace (or email us with it).

### UE5 logs

UE5 binary tends to print _tons_ of different logs to stdout. For that reason by default, we forward them to `/dev/null`. However, these may be useful if you are debugging weird errors in UE5. To fix that, you need to modify the `glimpse_generators/unreal_guardian.py` file by modifying this:

```python 
        self.process = subprocess.Popen(
            [str(self.unreal_binary_path), "-RenderOffscreen",  "-nosound"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
```

to this:

```python 
        self.process = subprocess.Popen(
            [str(self.unreal_binary_path), "-RenderOffscreen",  "-nosound"],
        )
```