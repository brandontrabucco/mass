import json
import glob
import os
import argparse
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

    parser.add_argument("--logdir", type=str, default="/home/btrabucco/detector-ablation")

    parser.add_argument("--keys", nargs='+', type=str, default=[
        "unshuffle/num_newly_misplaced",
        "unshuffle/prop_fixed_strict",
        "unshuffle/success"])

    args = parser.parse_args()

    all_json = glob.glob(os.path.join(args.logdir, "*/results/*.json"))
    with open(all_json[0], "r") as f:
        all_keys = [key for key, value in json.load(f).items()
                    if isinstance(value, int) or isinstance(value, float)]

    results = pd.DataFrame(columns=["Split", *all_keys])

    for json_name in all_json:

        with open(os.path.join(os.path.dirname(
                os.path.dirname(json_name)), "params-0-50.json"), "r") as f:

            params = json.load(f)

        with open(json_name, "r") as f:

            entries = {key: value for key, value
                       in json.load(f).items() if key in all_keys}

            entries.update(params)

            results = results.append(entries,
                                     ignore_index=True)

    methods = ["val", "test"]
    results = pd.concat([results[results['stage'] == method_name]
                         for method_name in methods])

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    fig, axs = plt.subplots(1, len(args.keys),
                            figsize=(10 * len(args.keys), 8))

    for i, key in enumerate(args.keys):

        axis = sns.lineplot(x="detection_threshold", y=key, hue="stage",
                            data=results, ci=68, linewidth=4,
                            ax=axs[i] if len(args.keys) > 1 else axs)

        axis.set(xlabel=None)
        axis.set(ylabel=None)

        if axis.get_legend() is not None:
            axis.get_legend().remove()

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')

        axis.yaxis.set_tick_params(labelsize=24)
        axis.xaxis.set_tick_params(labelsize=24)

        axis.set_xlabel("Detection Confidence Threshold", fontsize=36,
                        fontweight='bold', labelpad=12)

        axis.set_ylabel(pretty(key.replace("unshuffle/", "").replace("prop_fixed", "%Fixed")), fontsize=36,
                        fontweight='bold', labelpad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend([pretty(x.replace("val", "validation")) for x in methods],
                        loc="lower center", ncol=len(methods),
                        prop={'size': 36, 'weight': 'bold'})

    for i, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(sns.color_palette()[i])

    plt.tight_layout(pad=5.0)
    fig.subplots_adjust(bottom=0.35)

    plt.savefig(os.path.join(args.logdir,
                             "detector_ablation.pdf"))

    plt.savefig(os.path.join(args.logdir,
                             "detector_ablation.png"))

