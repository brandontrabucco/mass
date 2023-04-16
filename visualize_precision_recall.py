import json
import glob
import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--logdir", type=str,
                        default="/home/btrabucco/semantic-search-policy/policy-val/")
    parser.add_argument("--title", type=str, default="")

    args = parser.parse_args()

    all_df = [pd.read_csv(f, index_col=0) for f in
              glob.glob(os.path.join(args.logdir, "results/*.csv"))]

    sizes = pd.concat(all_df).groupby("type").mean()["size"]
    sizes = sizes.replace(to_replace=0, value=sizes.max()).sort_values()

    print(sizes)

    all_objects = []
    all_objects_recall = []
    all_objects_precision = []

    objects_moved_accuracy = pd.DataFrame(columns=["Object", "Success"])
    objects_to_move_accuracy = pd.DataFrame(columns=["Object", "Success"])

    for json_name in glob.glob(
            os.path.join(args.logdir, "results/*.json")):

        with open(json_name, "r") as f:

            data = json.load(f)

            if "unshuffle/objects_moved" not in data.keys() or \
                    "unshuffle/objects_to_move" not in data.keys():
                continue  # certain runs may not have logged this info

            for i, object_name in enumerate(data["unshuffle/objects_moved"]):

                if object_name not in all_objects:
                    all_objects.append(object_name)

                if all_objects_precision not in all_objects:
                    all_objects_precision.append(object_name)

                objects_moved_accuracy = objects_moved_accuracy.append(
                    dict(Object=object_name, Success=float(data[
                        "unshuffle/objects_moved_accuracy"][i])), ignore_index=True)

            for i, object_name in enumerate(data["unshuffle/objects_to_move"]):

                if object_name not in all_objects:
                    all_objects.append(object_name)

                if all_objects_recall not in all_objects:
                    all_objects_recall.append(object_name)

                objects_to_move_accuracy = objects_to_move_accuracy.append(
                    dict(Object=object_name, Success=float(data[
                        "unshuffle/objects_to_move_accuracy"][i])), ignore_index=True)

    aggregated = pd.concat([
        objects_moved_accuracy,
        objects_to_move_accuracy]).groupby("Object").mean()

    all_objects = [x for x in list(sizes.index) if x in all_objects]

    matplotlib.rc('font', family='Times New Roman', serif='cm10')
    matplotlib.rc('mathtext', fontset='cm')
    plt.rcParams['text.usetex'] = False

    fig, axs = plt.subplots(2, 1, figsize=(20, 10))

    axis = sns.barplot(x="Object", y="Success", linewidth=4, order=[
        n if n in all_objects_recall else "" for n in all_objects],
                       data=objects_to_move_accuracy, ci=68, ax=axs[0])

    axis.set(xlabel=None)
    axis.set(ylabel=None)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.xaxis.set_ticks_position('bottom')
    axis.yaxis.set_ticks_position('left')

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16, labelrotation=90)

    axis.set_ylabel("Recall", fontsize=24,
                    fontweight='bold', labelpad=12)

    axis.set_title(f"Which Map Differences (Small → Large) Are Detected{args.title}?",
                   fontsize=24, fontweight='bold', pad=12)

    axis.grid(color='grey', linestyle='dotted', linewidth=2)

    axis = sns.barplot(x="Object", y="Success", linewidth=4, order=[
        n if n in all_objects_precision else "" for n in all_objects],
                       data=objects_moved_accuracy, ci=68, ax=axs[1])

    axis.set(xlabel=None)
    axis.set(ylabel=None)

    axis.spines['right'].set_visible(False)
    axis.spines['top'].set_visible(False)

    axis.xaxis.set_ticks_position('bottom')
    axis.yaxis.set_ticks_position('left')

    axis.yaxis.set_tick_params(labelsize=16)
    axis.xaxis.set_tick_params(labelsize=16, labelrotation=90)

    axis.set_ylabel("Precision", fontsize=24,
                    fontweight='bold', labelpad=12)

    axis.set_title(f"Which Detections (Small → Large) Are Correct{args.title}?",
                   fontsize=24, fontweight='bold', pad=12)

    axis.grid(color='grey', linestyle='dotted', linewidth=2)

    plt.tight_layout(pad=3.0)

    plt.savefig(os.path.join(args.logdir,
                             "precision_recall.pdf"))

    plt.savefig(os.path.join(args.logdir,
                             "precision_recall.png"))
