import json
import glob
import os
import argparse
import numpy as np
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

    parser.add_argument("--logdir", type=str, default="/home/btrabucco/thor-analytics-val")
    parser.add_argument("--logdir2", type=str, default="/home/btrabucco/thor-analytics-test")

    parser.add_argument("--bins", type=int, default=10)

    parser.add_argument("--filter-need-rearrange", action="store_true")
    parser.add_argument("--filter-pickable", action="store_true")
    parser.add_argument("--filter-openable", action="store_true")

    parser.add_argument("--xs", nargs='+', type=str, default=[
        "size",
        "initial_distance",
        "initial_min_distance_type"])

    parser.add_argument("--ys", nargs='+', type=str, default=[
        "final_correct",
        "final_correct",
        "final_correct"])

    args = parser.parse_args()

    all_files = glob.glob(os.path.join(args.logdir, "*/results/*.csv"))
    all_df = [pd.read_csv(f, index_col=0) for f in all_files]

    methods = ["Validation", "Test"]

    for df, file_name in zip(all_df, all_files):

        method_name = os.path.basename(
            os.path.dirname(os.path.dirname(file_name)))

        stage = method_name.split("-")[-1]

        method_name = method_name.replace(
            "-test", "").replace("-val", "").replace("-train", "")

        df["Method"] = "Validation" if stage == "val" else "Test"

    results = pd.concat(all_df, ignore_index=True)

    results = pd.concat([results[results['Method'] == method_name]
                         for method_name in methods], ignore_index=True)

    if args.filter_need_rearrange:
        results = results[results["initial_correct"] == False]
    if args.filter_pickable:
        results = results[results["pickable"] == True]
    if args.filter_openable:
        results = results[results["openable"] == True]

    results["final_correct"] = results["final_correct"].astype(float)
    results["initial_correct"] = results["initial_correct"].astype(float)

    all_files = glob.glob(os.path.join(args.logdir2, "*/results/*.csv"))
    all_df = [pd.read_csv(f, index_col=0) for f in all_files]

    for df, file_name in zip(all_df, all_files):

        method_name = os.path.basename(
            os.path.dirname(os.path.dirname(file_name)))

        stage = method_name.split("-")[-1]

        method_name = method_name.replace(
            "-test", "").replace("-val", "").replace("-train", "")

        df["Method"] = "Validation" if stage == "val" else "Test"

    results2 = pd.concat(all_df, ignore_index=True)

    results2 = pd.concat([results2[results2['Method'] == method_name]
                         for method_name in methods], ignore_index=True)

    if args.filter_need_rearrange:
        results2 = results2[results2["initial_correct"] == False]
    if args.filter_pickable:
        results2 = results2[results2["pickable"] == True]
    if args.filter_openable:
        results2 = results2[results2["openable"] == True]

    results2["final_correct"] = results2["final_correct"].astype(float)
    results2["initial_correct"] = results2["initial_correct"].astype(float)

    results = pd.concat([results, results2], ignore_index=True)

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    fig, axs = plt.subplots(1, len(args.xs), figsize=(10 * len(args.xs), 8))

    for i, (x_key, y_key) in enumerate(zip(args.xs, args.ys)):

        filtered_results = results[results[x_key].notna()].copy()

        print(filtered_results)

        filtered_results[x_key] = pd.qcut(
            filtered_results[x_key], args.bins, duplicates='drop')

        filtered_results[x_key] = filtered_results[x_key].apply(
            lambda row: row.right)

        axis = sns.lineplot(x=x_key, y=y_key, hue="Method", data=filtered_results,
                            ci=68, linewidth=4,
                            ax=axs[i] if len(args.xs) > 1 else axs)

        axis.get_legend().remove()

        axis.set(xlabel=None)
        axis.set(ylabel=None)

        if i == 0:
            axis.set_xscale('log')

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')

        axis.yaxis.set_tick_params(labelsize=24)
        axis.xaxis.set_tick_params(labelsize=24)

        axis.set_xlabel(["Size (Meters³)",
                         "Distance To Goal (Meters)",
                         "Nearest Same Object (Meters)"][i], fontsize=36,
                        fontweight='bold', labelpad=12)

        if i == 0:
            axis.set_ylabel("%Fixed", fontsize=36,
                            fontweight='bold', labelpad=12)

        axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend(methods,
                        loc="lower center", ncol=len(methods),
                        prop={'size': 36, 'weight': 'bold'})

    for i, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(sns.color_palette()[i])

    plt.tight_layout(pad=5.0)
    fig.subplots_adjust(bottom=0.35)

    plt.savefig(os.path.join(args.logdir,
                             "analytics.pdf"))

    plt.savefig(os.path.join(args.logdir,
                             "analytics.png"))

    plt.savefig(os.path.join(args.logdir2,
                             "analytics.pdf"))

    plt.savefig(os.path.join(args.logdir2,
                             "analytics.png"))

