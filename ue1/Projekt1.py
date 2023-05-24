# Melf Fritsch
# Jakob Horbank
#
# Es wird erwartet, dass die es einen Ordner ./data gibt in dem die .wav Dateien liegen.
# Das IRMAS Archiv kann so wie es ist in den ./data Ordner entpackt werden.
#
# Abhängigkeiten installieren mit:
# pip install librosa scikit-learn tqdm

import os
import librosa
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


def get_cached_data():
    '''Load cached data dictionary. Returns none if cached is not found.'''
    file_dir = get_file_dir()
    fp = os.path.join(file_dir, "data", f"data.npz")
    try:
        data = np.load(fp)
    except :
        return None
    print(f"Loading cached data from {fp}")
    return data

def cache_data(data):
    '''Write data dictionary into .npz file'''
    file_dir = get_file_dir()
    fp = os.path.join(file_dir, "data", f"data.npz")
    if os.path.exists(fp):
        return
    
    np.savez(fp, **data)
    print(f"Data cached into {fp}")

def train_classifier(X_train, y_train):
    """Train Random Forest CLassifier on data and return training statistics"""

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2) # split into random train and validation sets
    classifier = RandomForestClassifier(250)

    classifier.fit(X_train, y_train)

    train_error = 1 - classifier.score(X_train, y_train) # score() uses the average of correct classifications by default
    val_error = 1 - classifier.score(X_val, y_val)

    return classifier, train_error, val_error

def make_plot(data, sr, clsnames):
    '''Create and save plot of used features'''
    signals = data['signals']
    labels = data['labels']
    num_signals = 4
    idxs = np.random.randint(0, len(signals), size=num_signals) # 4 random indices of signals

    fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True, sharey='row', figsize=(20,10)) # create 4x4 subplot

    for col, idx in enumerate(idxs):
        y=signals[idx]
        label=labels[idx]

        librosa.display.waveshow(y, sr=sr, axis='time', ax=ax[0, col])

        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr) # generate mel-spectrogram
        mel_spec_dB = librosa.power_to_db(mel_spec) # convert spectrogram values into dB scale                
        librosa.display.specshow(mel_spec_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax[1, col])

        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # calculate 13 mfcc values
        librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax[2, col])

        chroma = librosa.feature.chroma_stft(y=y, sr=sr) # calculate power spectrum chromagram
        librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax[3, col])

        ax[0, col].set_title(f"Instrument: {clsnames[label]}")
        ax[1, col].set_title("Mel Spectrogram")
        ax[2, col].set_title("MFCC")
        ax[3, col].set_title("Chromagram")

    fig.suptitle("Features for randomly picked signals (before averaging over time)")
    fig.tight_layout()

    fp = os.path.join(get_file_dir(), "features.jpg")
    fig.savefig(fp)
    print(f"Saved plot to {fp}")

def get_file_dir():
    return os.path.dirname(os.path.realpath(__file__))

def main():
    clsnames = ['gel', 'pia', 'voi', 'cel']
    sr = 44100

    # storage for all lists of features per signal
    data = {
        "signals": [],
        "labels": [],
        "mel_spectrogram_avg": [],
        "mfcc_avg": [],
        "chroma_avg": [],
        "combined": [],
    }

    # DATA LOADING/CACHING
    data_cached = get_cached_data()

    if not data_cached:
        print("No cached data found. Generating features...")

        data_dir = os.path.join(get_file_dir(), "data")
        files = librosa.util.find_files(data_dir, ext="wav", recurse=True) # recursive search for .wav files in data folder   

        instruments = [file.split("[", maxsplit=1)[1][:3] for file in files] # get main instrument for every file
        relevant_files = [(file, instrument) for file, instrument in zip(files, instruments) if instrument in clsnames] # get (filepath, instrument) pairs of wanted instruments

        for file, instrument in tqdm(relevant_files, unit="files"):
            label = clsnames.index(instrument) # convert instrument to label
            data['labels'].append(label)
            
            y, sr = librosa.load(file, sr=sr) # load wav and convert to mono by averaging stereo channels            
            
            all_features = []
            data['signals'].append(y)

            # FEATURE GENERATION
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr) # generate mel-spectrogram
            mel_spec_dB = librosa.power_to_db(mel_spec) # convert spectrogram values into dB scale
            mel_spec_dB_avg = np.mean(mel_spec_dB, axis=1) # average frequencies over time
            data['mel_spectrogram_avg'].append(mel_spec_dB_avg)
            all_features.append(mel_spec_dB_avg)
            
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # calculate 13 mfcc values
            mfcc_avg = np.mean(mfcc, axis=1) # average mfcc values over time
            data['mfcc_avg'].append(mfcc_avg)
            all_features.append(mfcc_avg)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr) # calculate chromagram
            chroma_avg = np.mean(chroma, axis=1) # average values over time
            data["chroma_avg"].append(chroma_avg)
            all_features.append(chroma_avg)

            # Merge sublists into a single list
            a = [j for i in all_features for j in i]
            data['combined'].append(a)
            
        cache_data(data)
                
    else:
        data = data_cached  

    make_plot(data, sr, clsnames)

    labels = data['labels']
    feature_list = list(data.items())[2:] # seperate features from signals and labels

    # TRAINING
    for feature_name, feature_data in feature_list:
        X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.05, random_state=1) # random_state to ensure same split for all tests

        print(f"\nFitting classifier on '{feature_name}' features...")
        classifier, train_error, val_error = train_classifier(X_train, y_train) # generate and train new classifier on current features
        print(f"{train_error=}")
        print(f"{val_error=}")

        test_error = 1 - classifier.score(X_test, y_test)
        print(f"{test_error=}")

if __name__ == "__main__":
    main()