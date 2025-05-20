import os
import csv
import numpy as np
import torchaudio
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import tensorflow as tf


def load_audio(filepath, target_sr=44100):
    """
    Load an audio file, convert to mono if stereo, and resample to target sample rate.
    """
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sr != target_sr:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)
        waveform = resampler(waveform)
    return waveform.squeeze(0).numpy().astype(np.float32), target_sr

#Computees db difference based on RMS
def compute_db_difference(wav1, wav2):
    """
    Compute average dB difference between two waveforms.
    """
    min_len = min(len(wav1), len(wav2))
    wav1 = wav1[:min_len]
    wav2 = wav2[:min_len]

    eps = 1e-10
    rms1 = np.sqrt(np.mean(wav1 ** 2) + eps)
    rms2 = np.sqrt(np.mean(wav2 ** 2) + eps)
    db_diff = 20 * np.log10(rms2 / rms1 + eps)
    return db_diff

# ##Based on clap
def compare_embeddings_clap(t1_np, tx_np):
    """
    Compare embeddings for the clap pipeline
    """
    ##Convert the embeddings to numpy 
    ##do the comparison on cpu instead of gpu
    # t1_np = t1_emb.detach().cpu().numpy()
    # tx_np = tx_emb.detach().cpu().numpy()

    ## https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html 
    mcss = np.max(cosine_similarity([t1_np], [tx_np])[0][0])
    
    ## euclid 
    euclid_dist = euclidean_distances([t1_np],[tx_np])[0][0]
    return mcss, euclid_dist

##Based on openl3
def compare_embeddings_openl3(t1_emb, tx_emb):
    cos_sim = np.max(cosine_similarity(t1_emb, tx_emb))

    t1_avg = np.mean(t1_emb, axis=0).reshape(1, -1)
    tx_avg = np.mean(tx_emb, axis=0).reshape(1, -1)

    euc_dist = euclidean_distances(t1_avg, tx_avg)[0][0]
    return cos_sim, euc_dist

def load_processed_songs(csv_path):
    processed = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if row:
                    processed.add((row[0], row[1]))
    return processed


def check_gpus():
    """
    Check GPU availability using TensorFlow (optional utility function).
    """

    # Show GPU availability
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print("Built with CUDA:", tf.test.is_built_with_cuda())
    print("GPUs:", tf.config.list_physical_devices('GPU'))

    # Ensure TensorFlow uses GPU if available
    gpus = tf.config.list_physical_devices('GPU')
    print("Num GPUs Available:", len(gpus))
    print(gpus)
