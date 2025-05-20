import os
import csv
import numpy as np
import torchaudio
import matplotlib.pyplot as plt
import pandas as pd




def compute_rms_db(waveform):
    eps = 1e-10
    rms = np.sqrt(np.mean(waveform ** 2) + eps)
    return 20 * np.log10(rms + eps)

## To mixed audio song vs individual stems average
def compare_mixture_vs_stems(root_folder, output_csv):
    stems = ["drums", "bass", "vocals", "other"]
    results = []

    for song_folder in sorted(os.listdir(root_folder)):
        song_path = os.path.join(root_folder, song_folder, "T1")
        if not os.path.isdir(song_path):
            continue

        song_name = os.path.basename(song_folder)
        print(f"Processing {song_name}")

        mix_path = os.path.join(song_path, "mixture.wav")
        if not os.path.exists(mix_path):
            print(f"  Skipping {song_name}: mixture.wav not found.")
            continue

        mix_audio, _ = load_audio(mix_path)
        mix_db = compute_rms_db(mix_audio)

        stem_dbs = {}
        for stem in stems:
            stem_path = os.path.join(song_path, f"{stem}.wav")
            if os.path.exists(stem_path):
                stem_audio, _ = load_audio(stem_path)
                stem_db = compute_rms_db(stem_audio)
                stem_dbs[stem] = stem_db

        if not stem_dbs:
            print(f"  Skipping {song_name}: no stems found.")
            continue

        avg_stem_db = np.mean(list(stem_dbs.values()))
        db_diff = mix_db - avg_stem_db

        entry = {
            "Song": song_name,
            "Mixture_dB": round(mix_db, 2),
            "Avg_Stem_dB": round(avg_stem_db, 2),
            "Difference_dB": round(db_diff, 2),
        }

        for stem in stems:
            stem_db = stem_dbs.get(stem)
            if stem_db is not None:
                entry[f"{stem}_dB"] = round(stem_db, 2)
                entry[f"Diff_{stem}"] = round(mix_db - stem_db, 2)
            else:
                # If stem is missing, use NaN or avg_stem_db as fallback
                entry[f"{stem}_dB"] = np.nan
                entry[f"Diff_{stem}"] = round(mix_db - avg_stem_db, 2)


        results.append(entry)

    # Save extended CSV
    fieldnames = (
    ["Song", "Mixture_dB", "Avg_Stem_dB", "Difference_dB"] +
    [f"{stem}_dB" for stem in stems] +
    [f"Diff_{stem}" for stem in stems]
    )

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSaved extended comparison CSV to: {output_csv}")



## Plots the difference between ground truth stems and mixture
## WOrks in conjunction with CSV produced by compare_mixture_vs_stems
def plot_mixture_vs_stems(csv_path, output_img_path):
    df = pd.read_csv(csv_path)

    if df.empty:
        print("CSV is empty, nothing to plot.")
        return

    stems = ["drums", "bass", "vocals", "other"]
    avg_total_diff = df["Difference_dB"].mean()
    stem_diffs = [df[f"Diff_{stem}"].mean() for stem in stems]

    labels = ["Avg All Stems"] + stems
    values = [avg_total_diff] + stem_diffs

    plt.figure(figsize=(8, 8))
    bars = plt.bar(labels, values, width= 0.6)
    
    plt.ylim(6, 7.5)

    plt.axhline(0, color='gray', linestyle='--')
    plt.ylabel("Average dB Difference (Mixture - Stem)")
    plt.title("Average dB Difference: Mixture vs Stems (MUSDB18HQ)")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f'{yval:.2f}', ha='center', va='bottom')

    print("\n=== dB Difference: Mixture vs Stems ===")
    for label, value in zip(labels, values):
        print(f"{label}: {value:.2f} dB")

    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()

    print(f"dB summary plot saved to: {output_img_path}")


## Plots the decibel difference between shifted tests (T2-4) in comparison to ground truth (T1)
def plot_db_difference(csv_path, output_img_path, overview_csv_path):
    stems = ["drums", "bass", "vocals", "other"]
    tests = ["T2", "T3", "T4"]

    df = pd.read_csv(csv_path)

    db_averages = []

    for stem in stems:
        db_row = []
        for test in tests:
            col = f"{test}_{stem}_db_diff"
            avg = df[col].mean() if col in df.columns else 0
            db_row.append(avg)
        db_averages.append(db_row)

    db_averages = np.array(db_averages)

    with open(overview_csv_path, "w", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Stem", "Test", "Avg_dB_Difference"])
        for stem_idx, stem in enumerate(stems):
            for test_idx, test in enumerate(tests):
                writer.writerow([
                    stem, test,
                    round(db_averages[stem_idx, test_idx], 2)
                ])

    x = np.arange(len(stems))
    width = 0.25

    print("\n=== dB Differences per Stem per Test ===")
    for stem_idx, stem in enumerate(stems):
        print(f"\nStem: {stem}")
        for test_idx, test in enumerate(tests):
            print(f"  {test}: {db_averages[stem_idx, test_idx]:.2f} dB")
            
    plt.figure(figsize=(8, 8))
    plt.bar(x - width, db_averages[:, 0], width, label='T2 Energy Bandwidth Following')
    plt.bar(x,         db_averages[:, 1], width, label='T3 STFT Shift       (<300Hz)')
    plt.bar(x + width, db_averages[:, 2], width, label='T4 Octave Shift     (-4 Oct)')
    plt.title("Average dB Difference from T1")
    plt.ylabel("dB Difference")
    plt.xticks(x, stems)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_img_path)
    plt.close()

    print(f"dB bar chart saved to: {output_img_path}")

## To compute dB between T1 and Tx stems for one song and write to CSV
## Is called in run dB comparisons which handles iteration
def process_song_folder_db(song_folder, stems, output_csv, processed_songs):
    song_name = os.path.basename(song_folder)
    t1_folder = os.path.join(song_folder, "T1")
    if not os.path.isdir(t1_folder):
        print(f"No T1 folder in {song_folder}")
        return

    song_data = {"Song": song_name}

    for stem in stems:
        t1_path = os.path.join(t1_folder, f"{stem}.wav")
        if not os.path.exists(t1_path):
            continue
        t1_audio, _ = load_audio(t1_path)

        for tx_name in sorted(os.listdir(song_folder)):
            if not tx_name.startswith("T") or tx_name == "T1":
                continue
            if (song_name, tx_name) in processed_songs:
                print(f"Skipping already processed: {song_name} | {tx_name}")
                continue

            tx_folder = os.path.join(song_folder, tx_name)
            tx_file = next((f for f in os.listdir(tx_folder) if f.lower().startswith(stem.lower()) and f.endswith(".wav")), None)
            if not tx_file:
                continue

            tx_path = os.path.join(tx_folder, tx_file)
            tx_audio, _ = load_audio(tx_path)

            db_diff = compute_db_difference(t1_audio, tx_audio)
            col_prefix = f"{tx_name}_{stem}"
            song_data[f"{col_prefix}_db_diff"] = db_diff
            print(f"{song_name} | {tx_name} | {stem} → ΔdB: {db_diff:.2f}")

    if len(song_data) > 1:
        write_header = not os.path.exists(output_csv)
        with open(output_csv, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=song_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(song_data)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Song row written (dB comparison).")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



## Iterates over the folders and stores dB differences
def run_db_comparisons(root_folder, output_csv):
    stems = ["drums", "bass", "vocals", "other"]
    processed_songs = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed_songs.add((row[0], row[1]))

    for song_folder in sorted(os.listdir(root_folder)):
        full_path = os.path.join(root_folder, song_folder)
        if os.path.isdir(full_path):
            process_song_folder_db(full_path, stems, output_csv, processed_songs)

# === Entry Point ===
if __name__ == "__main__":
    input_root = "../musdb18hq/merged/"
    output_csv = "../data/mixture_vs_stems_db.csv"
    output_plot = "../data/mixture_vs_stems_db_plot_tall.png"
    output_csv_db = "/home/dave/audio_split/clap_db_difference.csv"

##########################################################################
# Compares ground truth mixed audio to individual stems
# 
# Requires the file structure:
# /Database/Individual_Folder/
# ├── mixture.wav
# ├── drums.wav
# ├── bass.wav
# ├── vocals.wav
# └── other.wav
#
# Comment out the call to avoid rerunning the dB difference test.
##########################################################################


##########################################################################
    ## Plots ground truth difference
    # plot_mixture_vs_stems(output_csv, output_plot)

##########################################################################    


##########################################################################
    # Compare and plot shifted stem differences

    run_db_comparisons(input_root, output_csv_db)

    plot_db_difference(
        csv_path="../data/clap_db_difference.csv",
        output_img_path="../data/clap_db_barchart_tall.png",
        overview_csv_path="../data/clap_db_overview.csv"
    )

##########################################################################
