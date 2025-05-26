from __future__ import annotations
import argparse
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from dataset import GeneDataset
from classification import classifier_controller
from clustering import kmeans


def vote_and_map(cluster_labels: np.ndarray, true_labels: np.ndarray) -> dict[int, str]:
    mapping: dict[int, str] = {}
    for cid in np.unique(cluster_labels):
        members = cluster_labels == cid
        majority = Counter(true_labels[members]).most_common(1)[0][0]
        mapping[cid] = majority
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, default=0.9, help="probability threshold for one-vs-rest rejection")
    parser.add_argument("--k", type=int, default=2, help="k for K-Means clustering of unknowns")
    parser.add_argument("--seed", type=int, default=40, help="random seed for reproducibility")
    parser.add_argument("--show_cm", action="store_true", help="print confusion matrices")
    args = parser.parse_args()

    # Load data
    ds = GeneDataset()
    X_train, y_train = ds.get_train()
    X_test, y_test = ds.get_test()

    # Unique labels and prepare one-vs-rest classifiers
    labels = np.unique(y_train)
    ovr_clfs: dict[str, object] = {}
    for label in labels:
        clf = classifier_controller("svm")  # or choose 'svm', 'logreg', 'rf'
        # Fit binary: positive if equal to label, negative otherwise
        y_bin = (y_train == label).astype(int)
        clf.fit(X_train, y_bin)
        ovr_clfs[label] = clf

    # Compute one-vs-rest probabilities for test
    proba_mat = np.zeros((X_test.shape[0], labels.size), dtype=float)
    for idx, label in enumerate(labels):
        proba = ovr_clfs[label].predict_proba(X_test)[:, 1]
        proba_mat[:, idx] = proba

    # Decide known vs unknown
    known_idx = np.where(np.max(proba_mat, axis=1) >= args.theta)[0]
    unknown_idx = np.where(np.max(proba_mat, axis=1) < args.theta)[0]

    classifiers = ["one-vs-rest"]
    header = "#  Method       Clu.  Acc.   Prec.  Rec.   F1   Known  Unknown"
    print(header)
    print("-" * len(header))

    # For this single method
    y_pred = np.empty_like(y_test, dtype=object)
    clusters = 0

    # Predict known: choose label with highest prob
    if known_idx.size:
        best_idx = np.argmax(proba_mat[known_idx], axis=1)
        y_pred[known_idx] = labels[best_idx]

    # Cluster unknown samples
    if unknown_idx.size:
        X_unk = X_test[unknown_idx]
        clu = kmeans(X_unk, k=args.k, seed=args.seed)
        clusters = args.k
        mapping = vote_and_map(clu, y_test[unknown_idx])
        y_pred[unknown_idx] = np.array([mapping[c] for c in clu], dtype=object)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0)

    print(f"1   {'SVM (θ='+str(args.theta)+')':<14}{clusters:^6}"
          f"{acc:.3f} {prec:.3f} {rec:.3f} {f1:.3f} "
          f"{known_idx.size:^7}{unknown_idx.size:^9}")

    if args.show_cm:
        cm = confusion_matrix(y_test, y_pred)
        print("\n=== Confusion Matrix (One-vs-Rest) ===")
        print(cm)

if __name__ == '__main__':
    main()
