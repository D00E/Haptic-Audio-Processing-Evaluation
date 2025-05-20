import os
import csv
import openl3
import numpy as np
import torchaudio
import time
import gc
from tensorflow.keras import backend as K
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import euclidean
from concurrent.futures import ProcessPoolExecutor, as_completed

from audio_utils import load_audio, compute_db_difference, compare_embeddings_openl3, load_processed_songs, check_gpus

def batch_openl3_embeddings(audio_clips, sr=44100, embedding_size=512, hop_size=0.5):
    max_len = max(len(audio) for audio in audio_clips)
    padded_audio = [np.pad(audio, (0, max_len - len(audio))) for audio in audio_clips]
    embeddings, _ = openl3.get_audio_embedding(
        padded_audio,
        sr,
        content_type="music",
        embedding_size=embedding_size,
        frontend="kapre",
        hop_size=hop_size,
        batch_size=len(padded_audio)
    )
    # return [e.detach().cpu().numpy() for e in embeddings]
    return embeddings

def process_song_folder(song_folder, stems, output_csv, processed_songs):
    song_name = os.path.basename(song_folder)
    t1_folder = os.path.join(song_folder, "T1")
    if not os.path.isdir(t1_folder):
        print(f"No T1 folder in {song_folder}")
        return

    # Load T1 stems
    t1_audio = []
    valid_stems = []
    for stem in stems:
        path = os.path.join(t1_folder, f"{stem}.wav")
        if os.path.exists(path):
            audio, sr = load_audio(path)
            t1_audio.append(audio)
            valid_stems.append(stem)

    if not t1_audio:
        print(f"No valid stems in T1 of {song_folder}")
        return

    # Compute T1 embeddings in batch
    K.clear_session()
    gc.collect()
    t1_embeddings = batch_openl3_embeddings(t1_audio, sr=sr)
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

        tx_audio = []
        tx_stems = []
        for stem in valid_stems:
            tx_file = next((f for f in os.listdir(tx_folder) if f.lower().startswith(stem.lower()) and f.endswith(".wav")), None)
            if not tx_file:
                continue
            tx_path = os.path.join(tx_folder, tx_file)
            audio, _ = load_audio(tx_path)
            tx_audio.append(audio)
            tx_stems.append(stem)

        if not tx_audio:
            continue

        tx_embeddings = batch_openl3_embeddings(tx_audio, sr=sr)

        for stem, tx_emb in zip(tx_stems, tx_embeddings):
            t1_emb = t1_emb_dict[stem]
            cos_sim, euc_dist = compare_embeddings_openl3(t1_emb, tx_emb)
            col_prefix = f"{tx_name}_{stem}"
            song_data[f"{col_prefix}_cos"] = cos_sim
            song_data[f"{col_prefix}_euc"] = euc_dist
            print(f"{song_name} | {tx_name} | {stem} → Cosine: {cos_sim:.4f}, Euclidean: {euc_dist:.2f}")


    if len(song_data) > 1:  # more than just "Song" key
        write_header = not os.path.exists(output_csv)
        with open(output_csv, "a", newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=song_data.keys())
            if write_header:
                writer.writeheader()
            writer.writerow(song_data)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("Song row written.")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")



def run_openl3_comparisons(root_folder, output_csv):
    stems = ["drums", "bass", "vocals", "other"]
    processed_songs = load_processed_songs(output_csv)

    for song_folder in sorted(os.listdir(root_folder)):
        full_path = os.path.join(root_folder, song_folder)
        if os.path.isdir(full_path):
            start_song = time.time()
            process_song_folder(full_path, stems, output_csv, processed_songs)
            print(f"Time taken for {song_folder}: {time.time() - start_song:.2f}s")

def test_pipeline(root_folder):
    stems = ["drums", "bass", "vocals", "other"]
    song_folders = sorted([
        os.path.join(root_folder, d) for d in os.listdir(root_folder)
        if os.path.isdir(os.path.join(root_folder, d))
    ])[:3]  # Use first 3 folders for testing

    for song_folder in song_folders:
        song_name = os.path.basename(song_folder)
        t1_folder = os.path.join(song_folder, "T1")
        print(f"\nTesting self-similarity in {song_name}...")
        for stem in stems:
            path = os.path.join(t1_folder, f"{stem}.wav")
            if not os.path.exists(path):
                print(f"  Skipping {stem}: not found")
                continue

            audio, sr = load_audio(path)
            K.clear_session()
            gc.collect()
            emb = batch_openl3_embeddings([audio], sr=sr)[0]

            cos_sim, euc_dist = compare_embeddings_openl3(emb, emb)
            print(f"  {stem}: Cosine={cos_sim:.6f}, Euclidean={euc_dist:.6f}")

if __name__ == "__main__":
    check_gpus()
    input_root = "/home/dave/audio_split/musdb18hq/merged/"
    output_csv = "/home/dave/audio_split/openl3_similarity.csv"

    # start = time.time()
    # run_openl3_comparisons(input_root, output_csv)
    # end = time.time()
    # print(f"Time taken: {end - start:.2f} seconds")

    test_pipeline(input_root)