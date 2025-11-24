---
hide:
  - toc
  - navigation
---

<style>
.md-button {
    margin-bottom: .2rem;
}
</style>

<div style="text-align: center; margin-bottom: 3em;" markdown>
![Logo](imgs/logo.jpg){width="50%"}
<h1 style="margin-bottom: 0.1em !important;">FlySearch: Exploring how vision-language models explore</h1>

Adam Pardyl, Dominik Matuszek, Mateusz Przebieracz, Marek Cygan, Bartosz Zieliński, Maciej Wołczyk

NeurIPS 2025 Datasets & Benchmarks Track

[:material-trophy: Leaderboard](#leaderboard){ .md-button .md-button--primary } [:material-attachment: Paper](https://arxiv.org/pdf/2506.02896){ .md-button }  [:material-github: Code](https://github.com/gmum/FlySearch){ .md-button } [:material-presentation: Poster](imgs/poster.pdf){ .md-button }
</div>



!!! abstract "Tl;dr"
    A benchmark for evaluating vision-language models in simulated 3D, outdoor, photorealistic environments. Easy for humans, hard for state-of-the-art VLMs / MLLMs.

## Preview
Objective: locate a fire, environment: city, model: GPT-4o.

<video width="1920" height="1080" controls autoplay muted loop>
  <source src="imgs/preview.mp4" type="video/mp4">
</video>
<small>Video appears to be sped up due to slow frame capture rate.</small>

## Leaderboard


{% set df = pd_read_json('leaderboard.json') %}

{# Define groups: (Group label, [metric_keys...]) #}
{% set groups = [
  ("FS-1", ["fs-1-forest", "fs-1-city"]),
  ("FS-Anomaly-1", ["fs-anomaly-1-forest", "fs-anomaly-1-city"]),
  ("FS-2", ["fs-2-city"])
] %}

{# Define pretty names for metrics #}
{% set pretty = {
  "fs-1-forest": "Forest (%)",
  "fs-1-city":   "City (%)",
  "fs-anomaly-1-forest": "Forest (%)",
  "fs-anomaly-1-city":   "City (%)",
  "fs-2-city": "City (%)"
} %}

<div markdown style="overflow-x: auto;">
{{ render_leaderboard(df, groups, pretty) }}
</div>

The Progress Overall (%) metric refers to the average completion percentage of all FlySearch evaluation sets
(FS-1, FS-Anomaly, FS-2) by the model.

!!! tip "Join the challenge"
    If you would like to submit your agent to the leaderboard, please check the submission page.
    We accept submissions of both standard VLMs/MLLMs and agentic frameworks.
    
    [:material-invoice-text-send: Submit your agent](getting-started/30-submit.md){ .md-button .md-button--primary }

## Motivation: Vision-Language Models for Embodied AI exploration

Can Vision-Language Models (VLMs) perform active, goal-driven exploration of real-world environments?

* **The real world is messy and unstructured.** Uncovering critical information and decision-making requires
  curiosity, adaptability, and a goal-oriented mindset.
* **VLMs offer great zero-shot performance** in many difficult tasks ranging from image captioning to robotics.
* The ability of VLMs to operate in realistic, open-ended environments **remains largely untested**.

## Idea: Evaluate VLM exploration skills in 3D open world scenarios

**FlySearch** -- a new benchmark that evaluates exploration skills using vision-language reasoning.

* **Task:** locate an object specified in natural language or by visual examples.
* **Embodied interaction:** control an Unmanned Aerial Vehicle (UAV), observing images obtained from successive
  locations of the UAV and providing text commands that describe the next move.

<div style="text-align: center" markdown>
![FlySearch task](imgs/avebench.png){width="75%"}
</div>

## Realistic vision-language exploration benchmark

* **High-fidelity outdoor environment** built with Unreal Engine 5, enabling realistic and scalable evaluation of
  embodied agents in complex, unstructured settings.
* **A suite of object-based exploration challenges** designed to isolate and measure the
  exploration capabilities of VLMs and humans in open-world scenarios.
* **Procedurally generated.** We can generate an infinite number of vastly different test scenarios.
* **Easy to solve for a human, while challenging for popular VLMs.** We identify consistent failure
  modes across vision, grounding, and reasoning.

Our benchmark consists of two types of evaluation environments, forest
and city. For each environment, we can generate an infinite number of procedurally generated test
scenarios.

<div class="grid cards" markdown>
- :material-forest: __Forest environment__

    ---
    ![Forest](imgs/forest.jpg){width="80%"}

- :material-city: __City environment__

    ---
  ![City](imgs/city.jpg){width="80%"}

</div>

## Standardized evaluation set

We define three levels of difficulty of FlySearch and provide a test set of generated scenarios:

* **FS-1:** The target is visible from the starting position, but can be just a few pixels wide in the initial view. The
  object is described by text (e.g. "a red sport car").
* **FS-Anomaly-1:** The target object is an easy-to-spot anomaly (e.g. a flying saucer/UFO). The object description is
  _not_ given to the model, the task is to look for an anomaly. Other settings as in FS-1.
* **FS-2:** The object can be hidden behind buildings or be far away from the staring position. Additional visual
  preview of the object is given to the model at the start.

## Evaluation pipeline

FlySearch consists of three parts besides the vision-language model: the simulator, the evaluation controller, and the
scenario generator.

<div style="text-align: center" markdown>
![FlySearch architecture](imgs/arch.png){width="90%"}
</div>

## Benchmark environment details

* **Initial prompt:** Describes the target, provides information on how to perform actions, and the success conditions.
* **Observations:** During exploration, the model is provided with the current camera view of the UAV
  (500 × 500 px RGB image, downward-facing camera) and current altitude.
* **Action:** Relative movement coordinates or the _FOUND_ action (as text), preceded by an unlimited number of
  reasoning tokens.
* **Success condition:** The agent has the target's center in its view, is at most 10 meters above it, and responds with
  _FOUND_.

## More examples

Examples of a successful trajectories in FS-1 performed by GPT-4o. See more examples with complete agent-benchmark
conversations [here](/logs/).

<div style="text-align: center">
<video controls autoplay muted loop style="object-fit: cover; width:400px; height:400px">
  <source src="imgs/samples/6_hd.small.mp4" type="video/mp4">
</video>
<video controls autoplay muted loop style="object-fit: cover; width:400px; height:400px">
  <source src="imgs/samples/51_hd.small.mp4" type="video/mp4">
</video>
<video controls autoplay muted loop style="object-fit: cover; width:400px; height:400px">
  <source src="imgs/samples/100_hd.small.mp4" type="video/mp4">
</video>

<video controls autoplay muted loop style="object-fit: cover; width:400px; height:400px">
  <source src="imgs/samples/27_hd.small.mp4" type="video/mp4">
</video>
<video controls autoplay muted loop style="object-fit: cover; width:400px; height:400px">
  <source src="imgs/samples/108_hd.small.mp4" type="video/mp4">
</video>
<video controls autoplay muted loop style="object-fit: cover; width:400px; height:400px">
  <source src="imgs/samples/147_hd.small.mp4" type="video/mp4">
</video>
</div>


## Trajectory step by step
Example of a successful trajectory in FS-1 performed by GPT-4o. The agent navigates to red sports car
object by first descending and then moving to the right. The first row shows the model’s visual inputs,
and the second actions it has taken. Note the presence of the grid overlay on images.

<div style="text-align: center" markdown>
![Example trajectory](imgs/trajectory.png){width="75%"}
</div>



## Acknowledgements

<div style="text-align: justify; font-size: small">
This paper has been supported by the Horizon Europe Programme (HORIZONCL4-2022-HUMAN-02) under the project "ELIAS: European Lighthouse of AI for Sustainability", GA no. 101120237. This research was funded by National Science Centre, Poland (grant no. 2023/50/E/ST6/00469 and Sonata Bis grant no 2024/54/E/ST6/00388).
The research was supported by a grant from the Faculty of Mathematics and Computer Science under the
Strategic Programme Excellence Initiative at Jagiellonian University. We gratefully acknowledge Polish high-performance computing infrastructure PLGrid (HPC Center: ACK Cyfronet AGH) for providing computer facilities and support within computational grant no. PLG/2024/017483. Some experiments were performed on servers purchased with funds from the Priority Research Area (Artificial Intelligence Computing Center Core Facility) under the Strategic Programme Excellence Initiative at Jagiellonian University.
</div>

## Citation

{% raw %}
```
@inproceedings{pardyl2025flysearch,
  title        = {{FlySearch: Exploring how vision-language models explore}},
  author       = {Pardyl, Adam and Matuszek, Dominik and Przebieracz, Mateusz and Cygan, Marek and Zieliński, Bartosz and Wołczyk, Maciej},
  year         = 2025,
  booktitle    = {{Advances in Neural Information Processing Systems (NeurIPS)}},
  volume       = 39
}
```
{% endraw %}

## Contact

For questions, please open an issue on GitHub or contact Adam Pardyl -- adam.pardyl &lt;at&gt; doctoral.uj.edu.pl.

<div style="display: flex; flex-direction: row; flex-wrap: wrap; justify-content: space-between">
<a href="https://ideas-ncbr.pl/en/" rel="noreferrer" target="_blank"><img src="imgs/orgs/ideas.png" alt="IDEAS logo" style="height: 3em; margin-top: 0.6em;"/></a>
<a href="https://en.uj.edu.pl/en/" rel="noreferrer" target="_blank"><img src="imgs/orgs/uj.png" alt="UJ logo" style="height: 4em"/></a>
<a href="https://gmum.net/" rel="noreferrer" target="_blank"><img src="imgs/orgs/gmum.png" alt="GMUM logo" style="height: 4em"/></a>
<a href="https://en.uw.edu.pl/" rel="noreferrer" target="_blank"><img src="imgs/orgs/uw.png" alt="UW logo" style="height: 4em"/></a>
<a href="https://nomagic.ai/" rel="noreferrer" target="_blank"><img src="imgs/orgs/nomagic.png" alt="Nomagic logo" style="height: 4em"/></a>
</div>

<div style="text-align: center; margin: 2em 0" markdown>
[:material-attachment: Paper](https://arxiv.org/pdf/2506.02896){ .md-button } [:material-github: Code](https://github.com/gmum/FlySearch){ .md-button } [:material-rocket: Get started](getting-started/10-setup.md){ .md-button .md-button--primary }
</div>
