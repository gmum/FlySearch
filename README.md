# Active Visual GPT

This repository contains code for performing Active Visual Exploration Process on VLLMs.

## AVE benchmark

File `benchmark.py` contains example code for running an AVE process on a VLLM. To run it, install all dependencies and
simply run the following command:

```bash
python benchmark.py
```

It will create `all_logs` folder, within which a folder for a given run will be created. Variable `RUN_NAME` in the
script is responsible for the name of the folder.

To modify parameters of the AVE process, change the `benchmark.py` file.

### Change the model being used

You can change it by performing a dependency injection in the `VisualVStarExplorer` class of the `conversation`
argument. It should inherit from the `Conversation` class (or implement its interface). Currently, there are 3 models
available to be used:

- `OpenAIConversation` from the `openai_conversation.py` file
- `LLavaConversation` from the `llava_conversation.py` file
- `InternConversation` from the `intern_conversation.py` file

Due to the fact that these classes wrap around entirely different models, initializing them requires different
approaches. To see how to instantiate them properly, you can check main() functions in their respective files.

## Naive benchmark

To run the naive benchmark, run the following command:

```bash
python benchmark_simple.py
```

To modify model being benchmarked, you need to change code in the `benchmark_simple.py` file.

## Adding new models

To add support for new model, you need to create a new class that inherits from the `Conversation` class from
`abstract_conversation.py` file and implement
its interface.