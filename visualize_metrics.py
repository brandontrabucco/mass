import json
import glob
import os
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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", type=str, default="/home/btrabucco/slurm-data-v2")

    parser.add_argument("--keys", nargs='+', type=str, default=[
        "unshuffle/num_initially_misplaced",
        "unshuffle/prop_fixed_strict",
        "unshuffle/success"])

    args = parser.parse_args()

    all_json = glob.glob(os.path.join(args.logdir, "*/results/*.json"))
    with open(all_json[0], "r") as f:
        all_keys = [key for key, value in json.load(f).items()
                    if isinstance(value, int) or isinstance(value, float)]

    methods = list()
    results = pd.DataFrame(columns=["Method", *all_keys])

    for json_name in all_json:
        with open(json_name, "r") as f:

            entries = {key: value for key, value
                       in json.load(f).items() if key in all_keys}

            entries["Method"] = os.path.basename(
                os.path.dirname(os.path.dirname(json_name)))

            if entries["Method"] not in methods:
                methods.append(entries["Method"])

            results = results.append(entries,
                                     ignore_index=True)

    results = pd.concat([results[
        results['Method'] == method_name] for method_name in methods])

    groups = results.groupby("Method")
    sizes = groups.size()
    means = groups.mean()
    sem = groups.sem()

    prop_fixed_strict_mean = means["unshuffle/prop_fixed_strict"].to_numpy()
    prop_fixed_strict_sem = sem["unshuffle/prop_fixed_strict"].to_numpy()

    success_mean = means["unshuffle/success"].to_numpy()
    success_sem = sem["unshuffle/success"].to_numpy()

    prop_fixed_strict_sem *= scipy.stats.t.ppf((1 + 0.68) / 2., sizes.to_numpy() - 1)
    success_sem *= scipy.stats.t.ppf((1 + 0.68) / 2., sizes.to_numpy() - 1)

    print("unshuffle/prop_fixed_strict")
    for i, name in enumerate(means.index):
        print(name, prop_fixed_strict_mean[i], "\\pm", prop_fixed_strict_sem[i])

    print("unshuffle/success")
    for i, name in enumerate(means.index):
        print(name, success_mean[i], "\\pm", success_sem[i])

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    fig, axs = plt.subplots(1, len(args.keys),
                            figsize=(10 * len(args.keys), 8))

    for i, key in enumerate(args.keys):

        axis = sns.barplot(x="Method", y=key, data=results, ci=68,
                           linewidth=4,
                           ax=axs[i] if len(args.keys) > 1 else axs)

        axis.set(xlabel=None)
        axis.set(ylabel=None)

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')
        axis.xaxis.set_ticks([])

        axis.yaxis.set_tick_params(labelsize=16)
        axis.xaxis.set_tick_params(labelsize=16)

        axis.set_xlabel("Method", fontsize=24,
                        fontweight='bold', labelpad=12)

        axis.set_ylabel(key, fontsize=24,
                        fontweight='bold', labelpad=12)

        axis.set_title(pretty(key),
                       fontsize=24, fontweight='bold', pad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend([pretty(x) for x in methods],
                        loc="lower center", ncol=len(methods),
                        prop={'size': 24, 'weight': 'bold'})

    for i, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(sns.color_palette(n_colors=len(methods))[i])

    plt.tight_layout(pad=3.0)
    fig.subplots_adjust(bottom=0.2)

    plt.savefig(os.path.join(args.logdir,
                             "visualization.pdf"))

    plt.savefig(os.path.join(args.logdir,
                             "visualization.png"))

