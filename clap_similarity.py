import os
import csv
import numpy as np
import time
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.decomposition import PCA
from msclap import CLAP
import tensorflow as tf


from audio_utils import load_audio, compute_db_difference, compare_embeddings_clap, load_processed_songs, check_gpus


## Define 100 components for PCA
_pca = PCA(n_components=100)

def get_clap_embeddings(clap_model, file_paths, pca=None, aggregation='mean'):
    """
    Returns aggregated embeddings (mean or median) after optional PCA.
    """
    raw_embeddings = clap_model.get_audio_embeddings(file_paths)  # List of torch tensors
    aggregated_embeddings = []

    for emb in raw_embeddings:
        emb_np = emb.detach().cpu().numpy()
        if len(emb_np.shape) == 2:
            if aggregation == 'median':
                emb_np = np.median(emb_np, axis=0)
            else:
                emb_np = np.mean(emb_np, axis=0)
        if pca:
            emb_np = pca.transform([emb_np])[0]
        aggregated_embeddings.append(emb_np)

    return aggregated_embeddings




def process_song_folder(song_folder, stems, output_csv, processed_songs, clap_model):
    """
    Process a folder containing a song's original and transformed versions (T1, T2, etc.),
    extract embeddings, compare them, and write results to CSV.
    """
        
    song_name = os.path.basename(song_folder)
    t1_folder = os.path.join(song_folder, "T1")
    if not os.path.isdir(t1_folder):
        print(f"No T1 folder in {song_folder}")
        return

    t1_files = []
    valid_stems = []
    for stem in stems:
        path = os.path.join(t1_folder, f"{stem}.wav")
        if os.path.exists(path):
            t1_files.append(path)
            valid_stems.append(stem)

    if not t1_files:
        print(f"No valid stems in T1 of {song_folder}")
        return

    t1_embeddings = get_clap_embeddings(clap_model, t1_files)
    t1_emb_dict = dict(zip(valid_stems, t1_embeddings))

    song_data = {"Song": song_name}

    for tx_name in sorted(os.listdir(song_folder)):
        if not tx_name.startswith("T") or tx_name == "T1":
            continue
        if (song_name, tx_name) in processed_songs:
            print(f"Skipping already processed: {song_name} | {tx_name}")
            continue

        tx_folder = os.path.join(song_folder, tx_name)
        print(f"Processing {tx_name} in {song_name}")

        tx_files = []
        tx_stems = []
        for stem in valid_stems:
            tx_file = next((f for f in os.listdir(tx_folder) if f.lower().startswith(stem.lower()) and f.endswith(".wav")), None)
            if not tx_file:
                continue
            tx_path = os.path.join(tx_folder, tx_file)
            tx_files.append(tx_path)
            tx_stems.append(stem)

        if not tx_files:
            continue

        tx_embeddings = get_clap_embeddings(clap_model, tx_files)

        for stem, tx_emb in zip(tx_stems, tx_embeddings):
            t1_emb = t1_emb_dict[stem]
            cos_sim, euc_dist = compare_embeddings_clap(t1_emb, tx_emb)
            col_prefix = f"{tx_name}_{stem}"
            song_data[f"{col_prefix}_mcss"] = cos_sim
            song_data[f"{col_prefix}_euc"] = euc_dist
            print(f"{song_name} | {tx_name} | {stem} → MCSS: {cos_sim:.4f}, Euclidean: {euc_dist:.2f}")

    if len(song_data) > 1:
        write_header = not os.path.exists(output_csv)
        with open(output_csv, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=song_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(song_data)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Song row written.")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")


def run_clap_comparisons(root_folder, output_csv):
    """
    Main function to run CLAP-based embedding comparisons across all songs.
    """

    stems = ["drums", "bass", "vocals", "other"]
    processed_songs = set()
    if os.path.exists(output_csv):
        with open(output_csv, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed_songs.add((row[0], row[1]))

    clap_model = CLAP(version="2023", use_cuda=True)

    
    sample_paths = []
    for song_folder in sorted(os.listdir(root_folder)):
        t1_path = os.path.join(root_folder, song_folder, "T1")
        if os.path.isdir(t1_path):
            for f in os.listdir(t1_path):
                if f.endswith(".wav"):
                    sample_paths.append(os.path.join(t1_path, f))
                if len(sample_paths) >= 20:
                    break
        if len(sample_paths) >= 20:
            break

    # Get embeddings (list of torch tensors)
    raw_embs = clap_model.get_audio_embeddings(sample_paths)

    # Convert to list of numpy arrays (aggregated by mean across time if needed)
    raw_np = []
    for e in raw_embs:
        e = e.detach().cpu().numpy()
        if len(e.shape) == 2:  # (T, D)
            e = e.mean(axis=0)
        raw_np.append(e)

    # Ensure consistent shape before PCA
    raw_np = np.array(raw_np)

    # Fit PCA

    ##https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
    # pca_model = PCA(n_components=100).fit(raw_np)

    ## make sure theres aleast 100 samples :))))
    n_components = min(100, raw_np.shape[0], raw_np.shape[1])
    pca_model = PCA(n_components=n_components).fit(raw_np)
    print(f"number of components: {n_components}")

    for song_folder in sorted(os.listdir(root_folder)):
        full_path = os.path.join(root_folder, song_folder)
        if os.path.isdir(full_path):
            process_song_folder(full_path, stems, output_csv, processed_songs, clap_model)


def test_pipeline_clap(root_folder):
    """
    Self-similarity test to ensure embeddings pipeline is working.
    """
    stems = ["drums", "bass", "vocals", "other"]
    clap_model = CLAP(version="2023", use_cuda=True)

    song_folders = sorted([
        os.path.join(root_folder, d) for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ])[:3]

    for song_folder in song_folders:
        song_name = os.path.basename(song_folder)
        t1_folder = os.path.join(song_folder, "T1")
        print(f"\nTesting self-similarity in {song_name}...")
        file_paths = []
        valid_stems = []
        for stem in stems:
            path = os.path.join(t1_folder, f"{stem}.wav")
            if os.path.exists(path):
                file_paths.append(path)
                valid_stems.append(stem)

        if not file_paths:
            continue

        embeddings = get_clap_embeddings(clap_model, file_paths)
        for stem, emb in zip(valid_stems, embeddings):
            cos_sim, euc_dist = compare_embeddings_clap(emb, emb)
            print(f"  {stem}: MCSS={cos_sim:.6f}, Euclidean={euc_dist:.6f}")

def process_song_folder_db(song_folder, stems, output_csv, processed_songs):
    """
    Compare loudness (in dB) of stems between T1 and transformed versions.
    """

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

def run_db_comparisons(root_folder, output_csv):
    """
    Run dB difference comparisons across all stems.
    """

    stems = ["drums", "bass", "vocals", "other"]
    processed_songs = load_processed_songs(output_csv)

    for song_folder in sorted(os.listdir(root_folder)):
        full_path = os.path.join(root_folder, song_folder)
        if os.path.isdir(full_path):
            process_song_folder_db(full_path, stems, output_csv, processed_songs)



# === Entry Point ===
if __name__ == "__main__":
    check_gpus()
    input_root = "../musdb18hq/merged/"
    output_csv = "../data/clap_similarity.csv"
    output_csv_db = "../data/clap_db_difference.csv"

    # Uncomment for Main Clap MCSS embeddings Comparison
    start_time = time.time()
    run_clap_comparisons(input_root, output_csv)
    end_time = time.time()
    total_time = end_time - start_time

    print(f"\nTotal time taken for CLAP similarity comparison: {total_time:.2f} seconds")


    # Self-similarity test to test pipeline
    #Comment after 
    # test_pipeline_clap(input_root)


    # For dB difference comparisons
    # run_db_comparisons(input_root, output_csv_db)
