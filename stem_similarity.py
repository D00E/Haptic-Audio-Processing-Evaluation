import os
import csv
import time
import gc
import numpy as np
import torchaudio
import openl3
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from tensorflow.keras import backend as K

from audio_utils import load_audio, compute_db_difference, compare_embeddings_openl3, load_processed_songs, check_gpus

# === EMBEDDING ===
def get_openl3_embedding(audio, sr):
    try:
        emb, _ = openl3.get_audio_embedding(
            audio, sr,
            content_type="music",
            embedding_size=512,
            frontend="kapre",
            center=True,
            hop_size=1,
            batch_size=16
        )
        return emb
    except tf.errors.InternalError as e:
        print(f"GPU error during embedding: {e}")
        return None

# === SIMILARITY COMPARISON ===
def compare_tracks(t1_emb, tx_path, sr):
    audio2, sr2 = load_audio(tx_path, sr)
    emb2 = get_openl3_embedding(audio2, sr2)

    if emb2 is None or t1_emb is None:
        return 0.0, 0.0
    
    sim_matrix = cosine_similarity(t1_emb, emb2)  #(T1, T2)
    max_cos_sim = np.max(sim_matrix)  

    t1_mean = np.mean(t1_emb, axis=0)
    tx_mean = np.mean(emb2, axis=0)
    euclid_dist = euclidean(t1_mean, tx_mean)

    return max_cos_sim, euclid_dist

# === PROCESS A SINGLE SONG FOLDER ===
def process_song_folder(song_folder, stems, output_csv, processed_songs):
    song_name = os.path.basename(song_folder)
    t1_folder = os.path.join(song_folder, "T1")
    if not os.path.isdir(t1_folder):
        print(f"No T1 folder in {song_folder}")
        return

    t1_embeddings = {}
    K.clear_session()
    gc.collect()

    for stem in stems:
        t1_path = os.path.join(t1_folder, f"{stem}.wav")
        if not os.path.exists(t1_path):
            continue
        audio, sr = load_audio(t1_path)
        t1_embeddings[stem] = (get_openl3_embedding(audio, sr), sr)

    output_rows = []

    for tx_name in sorted(os.listdir(song_folder)):
        if not tx_name.startswith("T") or tx_name == "T1":
            continue
        if (song_name, tx_name) in processed_songs:
            print(f"Skipping already processed: {song_name} | {tx_name}")
            continue

        tx_folder = os.path.join(song_folder, tx_name)
        print(f"Processing {tx_name} in {song_name}")
        tx_files = os.listdir(tx_folder)

        for stem in stems:
            t1_emb, sr = t1_embeddings.get(stem, (None, None))
            if t1_emb is None:
                continue

            matching_files = [f for f in tx_files if f.lower().startswith(stem.lower()) and f.endswith(".wav")]
            if not matching_files:
                print(f"No match for {stem} in {tx_name}")
                continue

            tx_path = os.path.join(tx_folder, matching_files[0])
            try:
                audio2, sr2 = load_audio(tx_path, sr)
                tx_emb = get_openl3_embedding(audio2, sr2)

                if tx_emb is None or t1_emb is None:
                    cos_sim, euc_dist = 0.0, 0.0
                else:
                    cos_sim, euc_dist = compare_embeddings_openl3(t1_emb, tx_emb)

                output_rows.append([song_name, tx_name, stem, cos_sim, euc_dist])
                print(f"{song_name} | {tx_name} | {stem} → Cosine: {cos_sim:.4f}, Euclidean: {euc_dist:.2f}")
            except Exception as e:
                print(f"Error comparing {tx_path}: {e}")

    if output_rows:
        write_header = not os.path.exists(output_csv)
        with open(output_csv, "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["Song", "Test", "Stem", "OpenL3_Cosine", "OpenL3_Euclidean"])
            writer.writerows(output_rows)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("New row written :)")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

# === MAIN DRIVER ===
def run_openl3_comparisons(root_folder, output_csv):
    stems = ["drums", "bass", "vocals", "other"]
    processed_songs = load_processed_songs(output_csv)

    for song_folder in sorted(os.listdir(root_folder)):
        full_path = os.path.join(root_folder, song_folder)
        if os.path.isdir(full_path):
            start_song = time.time()
            process_song_folder(full_path, stems, output_csv, processed_songs)
            end_song = time.time()
            print(f"Elapsed time for {song_folder}: {end_song - start_song:.2f} sec")

# === RUN SCRIPT ===
if __name__ == "__main__":
    check_gpus()
    input_root = "../musdb18hq/merged/"
    output_csv = "../data/openl3_similarity.csv"

    start = time.time()
    run_openl3_comparisons(input_root, output_csv)
    end = time.time()

    print(f"\nTotal time taken: {end - start:.2f} seconds")
