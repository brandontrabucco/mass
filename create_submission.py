import json
import os
import glob
import gzip
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Create Submission")

    parser.add_argument("--train", type=str,
                        default="/home/btrabucco/slurm-data-v2/policy-train")
    parser.add_argument("--val", type=str,
                        default="/home/btrabucco/slurm-data-v2/policy-val")
    parser.add_argument("--test", type=str,
                        default="/home/btrabucco/slurm-data-v2/policy-test")
    parser.add_argument("--out", type=str,
                        default="submission.json.gz")

    args = parser.parse_args()

    submission = dict()

    files = (list(glob.glob(os.path.join(args.train, "results/*.json"))) +
             list(glob.glob(os.path.join(args.val, "results/*.json"))) +
             list(glob.glob(os.path.join(args.test, "results/*.json"))))

    lengths = []

    for metrics_file in files:

        with open(metrics_file, "r") as f:
            metrics = json.load(f)

        metrics.pop("unshuffle/objects_moved")
        metrics.pop("unshuffle/objects_moved_accuracy")
        metrics.pop("unshuffle/objects_to_move")
        metrics.pop("unshuffle/objects_to_move_accuracy")

        task_info = metrics.pop("task_info")
        submission[task_info["unique_id"]] = {**task_info, **metrics}

        if submission[task_info["unique_id"]]["walkthrough_actions"][-1] != "done" \
                and submission[task_info["unique_id"]]["walkthrough/ep_length"] < 250:

            submission[task_info["unique_id"]]["ep_length"] += 1
            submission[task_info["unique_id"]]["walkthrough/ep_length"] += 1

            submission[task_info["unique_id"]]["walkthrough_actions"].append("done")
            submission[task_info["unique_id"]]["walkthrough_action_successes"].append(True)

    print(len(submission))

    submission_str = json.dumps(submission)

    with gzip.open(args.out, "w") as f:
        f.write(submission_str.encode("utf-8"))
