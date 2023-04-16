import json
import glob
import os
import argparse
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


from mass.thor.segmentation_config import PICKABLE_TO_COLOR


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

    parser.add_argument("--logdir", type=str, default="/home/btrabucco/failure-modes")

    args = parser.parse_args()

    all_json = glob.glob(os.path.join(args.logdir, "*/results/*.json"))
    with open(all_json[0], "r") as f:
        all_keys = [key for key, value in json.load(f).items()
                    if isinstance(value, int) or isinstance(value, float)]

    methods = list()
    results = pd.DataFrame(columns=["Method", *all_keys])

    for json_name in all_json:

        task_id = os.path.basename(json_name).split(".")[0]

        analytics_df = pd.read_csv(os.path.join(
            os.path.dirname(json_name), f"analytics-{task_id}.csv"))

        with open(json_name, "r") as f:
            original_dict = json.load(f)
            entries = {key: value for key, value
                       in original_dict.items() if key in all_keys}

        entries["Method"] = os.path.basename(
            os.path.dirname(os.path.dirname(json_name)))

        entries["Split"] = original_dict["task_info"]["stage"].replace("val", "validation")

        if entries["Method"] not in methods:
            methods.append(entries["Method"])

        entries["TaskSolved"] = entries["unshuffle/success"]

        entries["ExceededTimeLimit"] = 1.0 if entries["unshuffle/success"] == 0.0 and \
            entries["unshuffle/ep_length"] == 500 else 0.0

        entries["IncorrectObjectRearranged"] = 1.0 if entries["unshuffle/success"] == 0.0 and \
            entries["ExceededTimeLimit"] == 0.0 and any([
                x not in original_dict["unshuffle/objects_to_move"]
                and not analytics_df[analytics_df["type"] == x]["final_correct"].all()
                for x in original_dict["unshuffle/objects_moved"]]) else 0.0

        entries["FailedToDetectDisagreement"] = 1.0 if entries["unshuffle/success"] == 0.0 and \
            entries["ExceededTimeLimit"] == 0.0 and \
            entries["IncorrectObjectRearranged"] == 0.0 and any([
                x not in original_dict["unshuffle/objects_moved"]
                for x in original_dict["unshuffle/objects_to_move"]]) else 0.0

        entries["FailedToRearrangeDisagreement"] = 1.0 if entries["unshuffle/success"] == 0.0 and \
            entries["ExceededTimeLimit"] == 0.0 and \
            entries["IncorrectObjectRearranged"] == 0.0 and \
            entries["FailedToDetectDisagreement"] == 0.0 else 0.0

        if entries["FailedToRearrangeDisagreement"] == 1.0:

            incorrect_objects = analytics_df.groupby("type").all()
            incorrect_objects = incorrect_objects[incorrect_objects["final_correct"] == False]
            incorrect_objects = list(incorrect_objects.index)

            print(task_id, entries["Method"])
            print(incorrect_objects)

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

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    keys = ["TaskSolved",
            "FailedToRearrangeDisagreement",
            "IncorrectObjectRearranged",
            "ExceededTimeLimit",
            "FailedToDetectDisagreement"]

    for ka, kb in zip(keys[:-1], keys[1:]):
        results[kb] = results[kb] + results[ka]

    fig, axs = plt.subplots(1, 2, figsize=(32, 8))

    for mi, m in enumerate(["sss", "policy"]):

        sub_results = results[results["Method"].str.contains(m)]

        for i, key in reversed(list(enumerate(keys))):

            axis = sns.barplot(x="Split", y=key, data=sub_results, ci=68,
                               linewidth=4, ax=axs[mi], color=sns.color_palette()[i])

            axis.set(xlabel=None)
            axis.set(ylabel=None)

            axis.spines['right'].set_visible(False)
            axis.spines['top'].set_visible(False)

            axis.xaxis.set_ticks_position('bottom')
            axis.yaxis.set_ticks_position('left')

            axis.yaxis.set_tick_params(labelsize=24)
            axis.xaxis.set_tick_params(labelsize=36)

            axis.set_ylabel("Proportion Of Tasks", fontsize=36,
                            fontweight='bold', labelpad=12)

            axis.set_title("Task Outcomes ({})".format("Ours + GT Both" if m == "sss" else "Ours"),
                           fontsize=36, fontweight='bold', pad=12)

            if i == len(keys) - 1:

                axis.grid(color='grey', linestyle='dotted', linewidth=2)

    legend = fig.legend([pretty(x) for x in keys],
                        loc="center right", ncol=1,
                        prop={'size': 36, 'weight': 'bold'})

    legend.set_title("Reasons For Task Failures",
                     prop={'size': 36, 'weight': 'bold'})

    for i, legend_object in enumerate(legend.legendHandles):
        legend_object.set_linewidth(4.0)
        legend_object.set_color(sns.color_palette()[i])

    plt.tight_layout(pad=5.0)
    fig.subplots_adjust(right=0.65, wspace=0.3)

    plt.savefig(os.path.join(args.logdir,
                             "failure_modes.pdf"))

    plt.savefig(os.path.join(args.logdir,
                             "failure_modes.png"))

