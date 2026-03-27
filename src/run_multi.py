import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
import os

from src.load_data import load_data
from src.preprocess import bandpass_filter
from src.model import train_svm
from src.evaluation import evaluate_full   # ✅ import once at top


# -------------------------
# REDUCE MNE LOGGING
# -------------------------
mne.set_log_level('WARNING')


# -------------------------
# CONFIG
# -------------------------
subjects = [
    "A01T.gdf",
    "A02T.gdf",
    "A03T.gdf",
    "A04T.gdf",
    "A05T.gdf",
    "A06T.gdf",
    "A07T.gdf",
    "A08T.gdf",
    "A09T.gdf",
]

data_path = "data/BCICIV_2a_gdf"


if __name__ == "__main__":

    accuracies = []

    # -------------------------
    # MAIN LOOP
    # -------------------------
    for subject in subjects:
        print("\n------------------------------")
        print(f"Processing {subject}")
        print("------------------------------")

        file_path = os.path.join(data_path, subject)

        try:
            raw = load_data(file_path)
            raw = bandpass_filter(raw)

            events, event_dict = mne.events_from_annotations(raw)

            event_id = {
                '769': event_dict['769'],
                '770': event_dict['770']
            }

            epochs = mne.Epochs(
                raw,
                events,
                event_id=event_id,
                tmin=1,
                tmax=4,
                baseline=None,
                preload=True
            )

            X = epochs.get_data()
            y = epochs.events[:, -1]

            y = y - min(y)

            print(f"Data shape: {X.shape}")

            # -------------------------
            # TRAIN MODEL
            # -------------------------
            accuracy = train_svm(X, y)

            print(f"Accuracy: {accuracy:.4f}")

            accuracies.append(accuracy)

            # -------------------------
            # ✅ EVALUATION METRICS (ADDED HERE)
            # -------------------------
            evaluate_full(X, y, subject.replace(".gdf", ""))

        except Exception as e:
            print(f"Error processing {subject}: {e}")
            accuracies.append(0)

    # -------------------------
    # CLEAN SUBJECT NAMES
    # -------------------------
    clean_subjects = [s.replace(".gdf", "") for s in subjects]

    # -------------------------
    # RESULTS
    # -------------------------
    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    for subj, acc in zip(clean_subjects, accuracies):
        print(f"{subj}: {acc:.4f}")

    avg_acc = np.mean(accuracies)
    print(f"\nAverage Accuracy: {avg_acc:.4f}")

    # -------------------------
    # STATISTICAL ANALYSIS
    # -------------------------
    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    median_acc = np.median(accuracies)
    cv = std_acc / mean_acc

    print("\nStatistical Summary")
    print(f"Mean Accuracy: {mean_acc:.4f}")
    print(f"Median Accuracy: {median_acc:.4f}")
    print(f"Std Deviation: {std_acc:.4f}")
    print(f"Min Accuracy: {min_acc:.4f}")
    print(f"Max Accuracy: {max_acc:.4f}")
    print(f"Coefficient of Variation: {cv:.4f}")
    print(f"Performance Gap: {(max_acc - min_acc):.4f}")

    # -------------------------
    # EXPORT TO CSV
    # -------------------------
    df = pd.DataFrame({
        "Subject": clean_subjects,
        "Accuracy": accuracies
    })

    df["Category"] = pd.cut(
        df["Accuracy"],
        bins=[0, 0.6, 0.7, 0.85, 1.0],
        labels=["Poor", "Moderate", "Good", "Excellent"]
    )

    df.to_csv("bci_results.csv", index=False)
    print("\nCSV file 'bci_results.csv' created successfully!")

    # -------------------------
    # CREATE RESULTS FOLDER
    # -------------------------
    os.makedirs("results", exist_ok=True)

    # -------------------------
    # SAVE PLOTS
    # -------------------------

    # Line plot
    plt.figure(figsize=(8, 5))
    plt.plot(clean_subjects, accuracies, marker='o', linewidth=2)
    plt.xlabel("Subjects")
    plt.ylabel("Accuracy")
    plt.title("Subject-wise EEG Classification Accuracy")
    plt.ylim(0, 1)
    plt.grid()
    plt.savefig("results/line_plot.png")
    plt.close()

    # Bar plot
    colors = [
        'red' if acc < 0.6 else
        'orange' if acc < 0.7 else
        'green'
        for acc in accuracies
    ]

    plt.figure(figsize=(8, 5))
    plt.bar(clean_subjects, accuracies, color=colors)

    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center')

    plt.xlabel("Subjects")
    plt.ylabel("Accuracy")
    plt.title("BCI Accuracy Across Subjects")
    plt.ylim(0, 1)
    plt.savefig("results/bar_plot.png")
    plt.close()

    # Box plot
    plt.figure(figsize=(5, 5))
    plt.boxplot(accuracies)
    plt.title("Accuracy Distribution Across Subjects")
    plt.ylabel("Accuracy")
    plt.savefig("results/box_plot.png")
    plt.close()

    print("Plots saved in 'results/' folder")
