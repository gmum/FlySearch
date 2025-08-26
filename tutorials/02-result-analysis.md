# Result analysis

Alright, you've run FlySearch on model of your choice. You are probably interested in actually analysing your model's performance. This tutorial part is meant to help you with doing that.

## Artifacts overview

 By this point, you should have a log folder that looks like this:
```
you_run_name
└───0
│   │   agent_info.json
│   │   conversation.json
│   │   scenario_params.json
│   │   simple_conversation.json
│   │   0.png
│   │   1.png
│   │   2.png
│   │   ...
│   │   0_coords.txt
│   │   1_coords.txt
│   │   2_coords.txt
│   │   ...
│   │   object_bbox.txt
│   │   termination.txt
│   
└───1
│   │   agent_info.json
│   │   conversation.json
│   │   scenario_params.json
│   │   simple_conversation.json
│   │   0.png
│   │   1.png
│   │   2.png
│   │   ...
│   │   0_coords.txt
│   │   1_coords.txt
│   │   2_coords.txt
│   │   ...
│   │   object_bbox.txt
│   │   termination.txt
│   
...
```

Each of these subfolders is a folder containing information about a singular trajectory. Saved images are enumerated images sent to the drone, while files with coordinates indicate agent's position _relative to the object being searched_. For example, if you see coordinates like `(0, 0, 100)` you know that the agent is directly above the target at 100 meters altitude. There is also a `scenario_params.json` which describes the scenario configuration and allows a replication of a trajectory via mimicking (see examples at `examples/mimic.sh` and documentation in `README.md` and `tutorials/00-script.md`). There are also `conversation.json` and `simple_conversation.json` showcasing agent's interactions with benchmark.

There is also `agent_info.json` where custom agents may save their internal state. However, if you were using the `simple_llm` agent, this file can be safely ignored.

## Generating a TeX document with trajectories

In the Appendix H of our [paper](https://arxiv.org/abs/2506.02896v2) we include example trajectories recorded in our benchmark. These allow to help researchers visually understand _how_ models are interacting with the environment in a way that pure numbers cannot convey that well. This subsection is meant to guide you through creating such a visualisation on your own.

To do it, locate the `analysis/report_generation/generate_appendix.py` file. It contains a `generate_report` function that we recommend using to create such a visualisation. 

Let's assume that you want to generate a report for model XYZ on FS-2 and that the run directory is `all_logs/XYZ-FS2`. You can then generate a TeX code for the run running a code like this:

```python 
with open("XYZ-FS2.tex", "w") as f:
    generate_report(
        model_displayname="XYZ",
        path_dir=pathlib.Path("../../all_logs/"),
        filter_func=lambda x: True,
        file=f,
        startfrom=0,
        n=200,
        subdir="XYZ-FS2",
        overwrite=True
    )
```

It will save the relevant TeX code to `XYZ-FS2.tex` and relevant images (figures and images that were sent to the agent) to `images/XYZ-FS2`. You can use it to add to your own TeX paper. If you _just_ want to have a standalone report, you can use an example TeX `main` file (`main.tex`). It looks like this:

```tex
\documentclass[12pt, a4paper]{report}

\usepackage[margin=0.8in]{geometry}
\usepackage{dramatist}
\usepackage{graphicx}
\usepackage{enumitem}
\usepackage{parskip}

\usepackage{setspace}
\onehalfspacing

\setcounter{secnumdepth}{3}

\begin{document}

\tableofcontents

\chapter{Examples}

\input{XYZ-FS2}

\end{document}
```

Note that you will need to change the name of the argument passed to the `\input` command to match name of the `.tex` file you've generated.

You can then compile the document by doing:

```bash 
set -e
pdflatex -interaction=nonstopmode -file-line-error -shell-escape --output-directory=build main.tex
pdflatex -interaction=nonstopmode -file-line-error -shell-escape --output-directory=build main.tex
```

or using premade script for that task (`compile.sh`):

```bash 
./compile.sh main
```