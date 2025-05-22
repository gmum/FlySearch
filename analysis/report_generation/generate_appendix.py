import os
import sys
import json
import re
import pathlib
import shutil

from PIL import Image
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, '../../')  # This is done so that we can import from the analysis module. # FIXME
from analysis import RunVisualiser, Run, RunAnalyser


def preamble(model_name):
    return """
    \\begin{drama}
    \\Character{Benchmark}{re}
    \\Character{GPT-4o}{gpt}
    """.replace("GPT-4o", model_name)


def one_glimpse_latex(glimpse_path):
    return f"""
    \\begin{'{center}'}
    \includegraphics[width=0.5\linewidth]{'{'}{glimpse_path}{'}'}
    \\end{'{center}'}
    """


def illustration_latex(illustration_path):
    return f"""
    \\begin{'{center}'}
    \includegraphics[width=0.7\linewidth]{'{'}{illustration_path}{'}'}
    \\end{'{center}'}
    """


def gpt_speaks(speech):
    return rf"\gptspeaks: {speech}".replace("_", "\\_").replace("<", "\( < \)").replace(">", "\( > \)")


def re_speaks(speech):
    return f"\\respeaks: {speech}".replace("_", "\\_").replace("<", "\( < \)").replace(">", "\( > \)")


def stage_direction():
    return "\StageDir{\\re{} presents \\gpt{} with a relevant image}"


def end_drama():
    return "\end{drama}"


def generate_section(name):
    return f"\\subsubsection{'{' + name + '}'}"


def generate_subsection(name):
    return f"\\subsection{'{' + name + '}'}"


def generate_metadata(run):
    starting_position = run.start_position
    final_position = run.end_position

    analyser = RunAnalyser(run)
    object_can_be_seen = analyser.object_visible()
    success = analyser.success_criterion_satisfied(10)

    try:
        object_type = run.get_params()["passed_object_name"]
    except KeyError:
        object_type = run.object_type

    return f"""
    \\begin{'{itemize}[noitemsep,topsep=0pt]'}
      \item Starting position: {starting_position}
      \item End position: {final_position} 
      \item Euclidean distance from the object: {(final_position[0] ** 2 + final_position[1] ** 2 + final_position[2] ** 2) ** 0.5}
      \item Object can be seen: {object_can_be_seen}
      \item Success: {success}
      \item Object type: {object_type}
      {'\item Username: ' + run.get_username() if run.username_recorded() else ''}
    \end{'{itemize}'}
    """.replace("_", "\\_")


def generate_report(model_name, model_displayname, env_name, path_dir, filter_func, n=2, overwrite=False,
                    file=sys.stdout, startfrom=0, subdir_override=None):
    subdir = subdir_override if subdir_override else f"{model_name}-{env_name}"
    image_dir = pathlib.Path("images") / subdir

    if overwrite:
        shutil.rmtree(image_dir, ignore_errors=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    run_dir = pathlib.Path(path_dir / subdir)

    fig_path = pathlib.Path("build/figures")

    if overwrite:
        shutil.rmtree(fig_path, ignore_errors=True)
    fig_path.mkdir(parents=True, exist_ok=True)

    hits = 0
    i = 0

    for one_run_dir in sorted([subdir for subdir in os.listdir(run_dir)], key=lambda x: int(x.split("_")[0])):
        one_run_dir = run_dir / str(one_run_dir)

        if i < startfrom:
            i += 1
            continue

        run = Run(one_run_dir)

        if not filter_func(run):
            continue

        hits += 1

        if hits > n:
            return

        run_visualiser = RunVisualiser(run)

        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)

        run_visualiser.plot(ax)

        fig.savefig(image_dir / f"{one_run_dir.name}.pdf")
        # fig.savefig(fig_path / f"{one_run_dir.name}.png")
        plt.close(fig)

        fig, ax = plt.subplots()
        run_visualiser.plot_situation_awareness_chart(ax)
        fig.savefig(image_dir / f"{one_run_dir.name}_sit_awareness.pdf")
        plt.close(fig)

        print(generate_section(f"Example"), file=file)

        print(generate_metadata(run), file=file)

        print(illustration_latex(image_dir / f"{one_run_dir.name}.pdf"), file=file)
        print(illustration_latex(image_dir / f"{one_run_dir.name}_sit_awareness.pdf"), file=file)

        preamble_name = run.username if run.username_recorded() else model_displayname

        print(preamble(model_name=preamble_name), file=file)

        # open simple conversation if exists, otherwise open conversation
        filename = "simple_conversation.json" if "simple_conversation.json" in os.listdir(one_run_dir) else "conversation.json"

        with open(one_run_dir / filename) as f:
            conversation = json.load(f)

            glimpse_count = 0

            for speaker, speech in conversation:
                if "user" in str(speaker).lower():
                    if speech != "image" and len(speech) < 20000:
                        if speech.strip().replace("\n", "").replace(" ", "").startswith("<Context>Youarein"):
                            #pass  # FIXME
                            speech = speech.split("<Objective>")[0].strip() + r" \texttt{(REST OF THE PROMPT)}"
                        print(re_speaks(speech), file=file)
                    else:
                        print(stage_direction(), file=file)
                        print(one_glimpse_latex(image_dir / f"{one_run_dir.name}_{glimpse_count}.jpg"), file=file)
                        Image.open(one_run_dir / f"{glimpse_count}.png").save(
                            image_dir / f"{one_run_dir.name}_{glimpse_count}.jpg")

                        #shutil.copyfile(one_run_dir / f"{glimpse_count}.png",
                        #                image_dir / f"{one_run_dir.name}_{glimpse_count}.png")
                        glimpse_count += 1
                else:
                    print(gpt_speaks(speech), file=file)

        print(end_drama(), file=file)


def main():

    # GPT-4o, FS1, City

    with open("GPT4o-City-Success-FS1.tex", "w") as f:
        generate_report("GPT-4o", "GPT-4o", "CityNew", pathlib.Path("../../all_logs/"),
                        lambda x: RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="GPT4o-CityNew",
                        overwrite=True)

    with open("GPT4o-City-Failure-FS1.tex", "w") as f: 
        generate_report("GPT-4o", "GPT-4o", "CityNew", pathlib.Path("../../all_logs/"),
                        lambda x: not RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="GPT4o-CityNew",
                        overwrite=False)
        
    # GPT-4o, FS1, Forest
        
    with open("GPT4o-Forest-Success-FS1.tex", "w") as f:
        generate_report("GPT-4o", "GPT-4o", "ForestNew", pathlib.Path("../../all_logs/"),
                        lambda x: RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="GPT4o-ForestNew",
                        overwrite=True)

    with open("GPT4o-Forest-Failure-FS1.tex", "w") as f: 
        generate_report("GPT-4o", "GPT-4o", "ForestNew", pathlib.Path("../../all_logs/"),
                        lambda x: not RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="GPT4o-ForestNew",
                        overwrite=False)
        

    # GPT-4o, FS2

    with open("GPT4o-City-Success-FS2.tex", "w") as f:
        generate_report("GPT-4o", "GPT-4o", "CityNew", pathlib.Path("../../all_logs/"),
                        lambda x: RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="GPT4o-FS2-City-Reckless1",
                        overwrite=True)

    with open("GPT4o-City-Failure-FS2.tex", "w") as f:
        generate_report("GPT-4o", "GPT-4o", "CityNew", pathlib.Path("../../all_logs/"),
                        lambda x: not RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="GPT4o-FS2-City-Reckless1",
                        overwrite=False)
    
    # Claude 3.5, FS1, City 

    with open("Claude-City-Success-FS1.tex", "w") as f:
        generate_report("Claude 3.5 Sonnet", "Claude 3.5 Sonnet", "CityNew", pathlib.Path("../../all_logs/"),
                        lambda x: RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="Sonnet-CityNew",
                        overwrite=True)

    with open("Claude-City-Failure-FS1.tex", "w") as f: 
        generate_report("Claude 3.5 Sonnet", "Claude 3.5 Sonnet", "CityNew", pathlib.Path("../../all_logs/"),
                        lambda x: not RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="Sonnet-CityNew",
                        overwrite=False)
        
    # Claude 3.5, FS1, Forest
        
    with open("Claude-Forest-Success-FS1.tex", "w") as f:
        generate_report("Claude 3.5 Sonnet", "Claude 3.5 Sonnet", "ForestNew", pathlib.Path("../../all_logs/"),
                        lambda x: RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="Sonnet-ForestNew",
                        overwrite=True)

    with open("Claude-Forest-Failure-FS1.tex", "w") as f: 
        generate_report("Claude 3.5 Sonnet", "Claude 3.5 Sonnet", "ForestNew", pathlib.Path("../../all_logs/"),
                        lambda x: not RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="Sonnet-ForestNew",
                        overwrite=False)

    # Claude 3.5, FS2    
    
    with open("Claude-City-Success-FS2.tex", "w") as f:
        generate_report("Claude 3.5 Sonnet", "Claude 3.5 Sonnet", "CityNew", pathlib.Path("../../all_logs/"),
                        lambda x: RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="Claude35-FS2-Reckless1",
                        overwrite=True)

    with open("Claude-City-Failure-FS2.tex", "w") as f:
        generate_report("Claude 3.5 Sonnet", "Claude 3.5 Sonnet", "CityNew", pathlib.Path("../../all_logs/"),
                        lambda x: not RunAnalyser(x).success_criterion_satisfied(), file=f, startfrom=20, n=2, subdir_override="Claude35-FS2-Reckless1",
                        overwrite=False)
        
    
if __name__ == "__main__":
    main()
