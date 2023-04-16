import json
import glob
import os
import numpy as np
import argparse
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


def pretty(text):
    """Convert a string into a consistent format for
    presentation in a matplotlib pyplot:
    this version looks like: One Two Three Four
    """

    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = text.replace("/", " ")
    text = text.strip()
    prev_c = None
    out_str = []
    for c in text:
        if prev_c is not None and \
                prev_c.islower() and c.isupper():
            out_str.append(" ")
            prev_c = " "
        if prev_c is None or prev_c == " ":
            c = c.upper()
        out_str.append(c)
        prev_c = c
    return "".join(out_str)


def load_npy_files(args):

    data = []
    max_length = 0

    for f in glob.glob(os.path.join(args.logdir, "*walkthrough*/results/objects-found-walkthrough-*.npy")):

        x = (np.load(f) < args.threshold).astype(np.float32).cumsum(axis=1).clip(max=1).mean(axis=0) * 100

        data.append(x)
        max_length = max(max_length, x.size)

    data = [np.concatenate([x, np.full([max_length - x.size], x[-1])]) for x in data]

    records = [
        dict(timestep=t, found=x[t if t < x.size else -1], method="Semantic Search", phase="Walkthrough")
        for x in data for t in range(max_length)
    ]

    data = []
    max_length = 0

    for f in glob.glob(os.path.join(args.logdir, "*unshuffle*/results/objects-found-unshuffle-*.npy")):

        x = (np.load(f) < args.threshold).astype(np.float32).cumsum(axis=1).clip(max=1).mean(axis=0) * 100

        data.append(x)
        max_length = max(max_length, x.size)

    data = [np.concatenate([x, np.full([max_length - x.size], x[-1])]) for x in data]

    records += [
        dict(timestep=t, found=x[t if t < x.size else -1], method="Semantic Search", phase="Unshuffle")
        for x in data for t in range(max_length)
    ]

    data = []
    max_length = 0

    for f in glob.glob(os.path.join(args.logdir, "no-semantic-search*/results/objects-found-walkthrough-*.npy")):

        x = (np.load(f) < args.threshold).astype(np.float32).cumsum(axis=1).clip(max=1).mean(axis=0) * 100

        data.append(x)
        max_length = max(max_length, x.size)

    data = [np.concatenate([x, np.full([max_length - x.size], x[-1])]) for x in data]

    records += [
        dict(timestep=t, found=x[t if t < x.size else -1], method="Uniform Baseline", phase="Walkthrough")
        for x in data for t in range(max_length)
    ]

    data = []
    max_length = 0

    for f in glob.glob(os.path.join(args.logdir, "no-semantic-search*/results/objects-found-unshuffle-*.npy")):

        x = (np.load(f) < args.threshold).astype(np.float32).cumsum(axis=1).clip(max=1).mean(axis=0) * 100

        data.append(x)
        max_length = max(max_length, x.size)

    data = [np.concatenate([x, np.full([max_length - x.size], x[-1])]) for x in data]

    records += [
        dict(timestep=t, found=x[t if t < x.size else -1], method="Uniform Baseline", phase="Unshuffle")
        for x in data for t in range(max_length)
    ]

    return pd.DataFrame.from_records(records)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", type=str, default="/home/btrabucco/slurm-data-v2")

    parser.add_argument("--threshold", type=float, default=1.0)

    args = parser.parse_args()

    results = load_npy_files(args)

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    fig, axs = plt.subplots(1, 2, figsize=(10 * 2, 6))

    for i, phase in enumerate(["Walkthrough", "Unshuffle"]):

        selected_results = results[results["phase"] == phase]

        axis = sns.lineplot(x="timestep", y="found", hue="method", 
                            data=selected_results, ci=68, linewidth=4, ax=axs[i])

        axis.set(xlabel=None)
        axis.set(ylabel=None)

        if axis.get_legend() is not None:
            axis.get_legend().remove()

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')

        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel("Episode Timestep", fontsize=24,
                        fontweight='bold', labelpad=12)

        axis.set_ylabel("% Objects Found (Test)", fontsize=24,
                        fontweight='bold', labelpad=12)

        axis.set_title(f"{phase} Phase",
                       fontsize=24, fontweight='bold', pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)


        for timestep_slice in [100, 250, 500]:

            ss_results = selected_results[(selected_results["timestep"] == timestep_slice) & (
                selected_results["method"] == "Semantic Search")]["found"].to_numpy()

            uni_results = selected_results[(selected_results["timestep"] == timestep_slice) & (
                selected_results["method"] == "Uniform Baseline")]["found"].to_numpy()

            improvement = ss_results - uni_results
            improvement_sem = scipy.stats.sem(improvement) * scipy.stats.t.ppf((1 + 0.68) / 2., improvement.size - 1)

            print(phase, timestep_slice, improvement.mean(), "\\pm", improvement_sem)


    legend = fig.legend(["Semantic Search", "Uniform Baseline"],
                        loc="lower center", ncol=2,
                        prop={'size': 24, 'weight': 'bold'})

    for i, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(sns.color_palette(n_colors=2)[i])

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(bottom=0.3)

    plt.savefig(os.path.join(args.logdir,
                             "found_objects.pdf"))

    plt.savefig(os.path.join(args.logdir,
                             "found_objects.png"))

