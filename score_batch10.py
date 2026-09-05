"""Score a submission CSV against the held-out true labels in data/batch10.dat.

batch10.dat is UCI gas-sensor-drift format: "class;concentration idx:val idx:val ...",
one line per measurement, in the same order as the batch-10 rows of test.csv.
We recover which measurement_id each line belongs to from that row order, then
compare each id's true class to the submission's predicted class.

Usage: python score_batch10.py <submission.csv> [--batch10 data/batch10.dat] [--test data/test.csv]
"""
import argparse
import pandas as pd
from sklearn.metrics import f1_score, classification_report, confusion_matrix

def true_labels(batch10_path, test_path, batch_num=10):
    labels = [int(line.split(";", 1)[0]) for line in open(batch10_path)]
    ids = pd.read_csv(test_path)
    ids = ids.loc[ids["batch"] == batch_num, "measurement_id"].tolist()
    assert len(ids) == len(labels), f"row count mismatch: {len(ids)} ids vs {len(labels)} labels"
    return dict(zip(ids, labels))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--batch10", default="data/batch10.dat")
    ap.add_argument("--test", default="data/test.csv")
    args = ap.parse_args()

    truth = true_labels(args.batch10, args.test)
    sub = pd.read_csv(args.submission).set_index("measurement_id")["gas_class"]

    missing = [i for i in truth if i not in sub.index]
    if missing:
        raise SystemExit(f"submission is missing {len(missing)} ids, e.g. {missing[:5]}")

    y_true = [truth[i] for i in truth]
    y_pred = [int(sub[i]) for i in truth]

    classes = sorted(set(y_true) | set(y_pred))
    f1_macro = f1_score(y_true, y_pred, labels=classes, average="macro")

    print(f"F1_macro = {f1_macro:.4f}\n")
    print(classification_report(y_true, y_pred, labels=classes, digits=4))
    print("confusion matrix (rows=true, cols=pred):")
    print(pd.DataFrame(confusion_matrix(y_true, y_pred, labels=classes), index=classes, columns=classes))

if __name__ == "__main__":
    main()
