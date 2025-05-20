import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import csv

def plot_mcss_and_euclidean(csv_path, output_mcss_path, output_euc_path, overview_csv_path):
    df = pd.read_csv(csv_path)

    stems = ["drums", "bass", "vocals", "other"]
    tests = ["T2", "T3", "T4"]

    cosine_averages = []
    euclidean_averages = []

    for stem in stems:
        cos_row = []
        euc_row = []
        for test in tests:
            cos_col = f"{test}_{stem}_mcss"
            euc_col = f"{test}_{stem}_euc"

            cos_avg = df[cos_col].mean() if cos_col in df.columns else 0
            euc_avg = df[euc_col].mean() if euc_col in df.columns else 0

            cos_row.append(cos_avg)
            euc_row.append(euc_avg)

        cosine_averages.append(cos_row)
        euclidean_averages.append(euc_row)

    cosine_averages = np.array(cosine_averages)
    euclidean_averages = np.array(euclidean_averages)

    # Save overview CSV
    with open(overview_csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Stem", "Test", "Avg_Cosine", "Avg_Euclidean"])
        for stem_idx, stem in enumerate(stems):
            for test_idx, test in enumerate(tests):
                writer.writerow([
                    stem, test,
                    round(cosine_averages[stem_idx, test_idx], 2),
                    round(euclidean_averages[stem_idx, test_idx], 2)
                ])

    # Plot 1: MCSS
    x = np.arange(len(stems))
    width = 0.25

    print("\n=== MCSS and Euclidean Averages ===")
    for stem_idx, stem in enumerate(stems):
        print(f"\nStem: {stem}")
        for test_idx, test in enumerate(tests):
            cos_score = round(cosine_averages[stem_idx, test_idx], 4)
            euc_score = round(euclidean_averages[stem_idx, test_idx], 4)
            print(f"  {test}: MCSS = {cos_score}, Euclidean = {euc_score}")

    plt.figure(figsize=(15, 12))
    plt.bar(x - width, cosine_averages[:, 0], width, label='T2 Energy Bandwidth Following')
    plt.bar(x,         cosine_averages[:, 1], width, label='T3 STFT Shift       (<300Hz)')
    plt.bar(x + width, cosine_averages[:, 2], width, label='T4 Octave Shift     (-4 Oct)')
    plt.title("Average MCSS Similarity",fontsize = 16)
    plt.ylabel("MCSS Similarity",fontsize = 14)
    plt.xticks(x, stems,fontsize = 14)
    plt.legend(fontsize = 12)
    # plt.tight_layout()
    plt.savefig(output_mcss_path, dpi = 300)
    plt.close()
    print(f"MCSS bar chart saved to: {output_mcss_path}")

    # Plot 2: Euclidean
    plt.figure(figsize=(15, 12))
    plt.bar(x - width, euclidean_averages[:, 0], width, label='T2 Energy Bandwidth Following')
    plt.bar(x,         euclidean_averages[:, 1], width, label='T3 STFT Shift       (<300Hz)')
    plt.bar(x + width, euclidean_averages[:, 2], width, label='T4 Octave Shift     (-4 Oct)')
    plt.title("Average Euclidean Distance", fontsize = 16)
    plt.ylabel("Euclidean Distance",fontsize = 14)
    plt.xticks(x, stems,fontsize = 14)
    # plt.legend(fontsize = 10)
    plt.legend(fontsize=12, loc='upper right', bbox_to_anchor=(1, 1))
    # plt.tight_layout()
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_euc_path, dpi=300)
    plt.show()
    plt.close()
    print(f"Euclidean bar chart saved to: {output_euc_path}")



if __name__ == "__main__":
    # To Plot MCSS and Euclidean distance

    plot_mcss_and_euclidean(
        csv_path="../data/clap_similarity.csv",
        output_mcss_path="../data/clap_mcss_barchart_large.png",
        output_euc_path="../data/clap_euc_barchart_large.png",
        overview_csv_path="../data/clap_overview.csv"
    )
