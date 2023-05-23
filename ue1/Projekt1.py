# Melf Fritsch
# Jakob Horbank
#
# Es wird erwartet, dass die es einen Ordner ./data gibt, in dem vier Ordner gel, pia, voi, cel mit den jeweiligen .wav Dateien liegen.
# Das Skript und der ./data Ordner müssen im cwd liegen
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

data_dir = os.path.join(os.getcwd(), "data")
clsnames = ['gel', 'pia', 'voi', 'cel']

sr = 44100

# # for debugging purposes
# labels = [1]
# features = {
#     "mel_spectrogram_avg": [np.array([1,2,3,4,5])],
#     "mfcc_avg": [np.array([1,1,1,1,1])],
#     "chroma_avg": [np.array([1,1,1,1,1])],
# }

labels = []

# storage for all lists of features per signal
features = {
    "mel_spectrogram_avg": [],
    "mfcc_avg": [],
    "chroma_avg": [],
}

def train_classifier(X_train, y_train):
    """Train Random Forest CLassifier on data and return training statistics"""

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2) # split into random train and validation sets
    classifier = RandomForestClassifier(100)

    classifier.fit(X_train, y_train)

    train_error = 1 - classifier.score(X_train, y_train) # score() uses the average of correct classifications by default
    val_error = 1 - classifier.score(X_val, y_val)

    return classifier, train_error, val_error

fig, ax = plt.subplots(nrows=4, ncols=4, sharex=True) # create 4x4 subplot

# DATA LOADING
print("Reading data and generating features")
for label, clsname in enumerate(clsnames):
    clsdir = os.path.join(data_dir, clsname)

    for i, file in enumerate(tqdm(os.listdir(clsdir), desc=f"{clsname}", unit="files")):
        if file.endswith('.wav'):
            filepath = os.path.join(clsdir, file)
            y, sr = librosa.load(filepath) # load wav and convert to mono by averaging stereo channels
            
            # FEATURE GENERATION
            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr) # generate mel-spectrogram
            mel_spec_dB = librosa.power_to_db(mel_spec) # convert spectrogram values into dB scale
            mel_spec_dB_avg = np.mean(mel_spec_dB, axis=1) # average frequencies over time
            features['mel_spectrogram_avg'].append(mel_spec_dB_avg)

            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13) # calculate 13 mfcc values
            mfcc_avg = np.mean(mfcc, axis=1) # average mfcc values over time
            features['mfcc_avg'].append(mfcc_avg)

            chroma = librosa.feature.chroma_stft(y=y, sr=sr) # calculate power spectrum chromagram
            chroma_avg = np.mean(chroma, axis=1) # average values over time
            features["chroma_avg"].append(chroma_avg)

            labels.append(label) 

            # plot signal and features (before averaging) for first example from every class
            if i == 0:                
                ax[0,label].set_title(f"Instrument: {clsname}")
                librosa.display.waveshow(y, sr=sr, axis='time', ax=ax[0, label])                
                librosa.display.specshow(mel_spec_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax[1, label])
                librosa.display.specshow(mfcc, sr=sr, x_axis="time", ax=ax[2, label])
                librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax[3, label])

                
labels = np.array(labels)
print("Done.")       

# PLOT
print("If the plot is ugly make it bigger :)")
print("Close plot to start training")

ax[0,0].set(ylabel="Signal")
ax[1,0].set(ylabel="Mel Spectrogram")
ax[2,0].set(ylabel="MFCC")
ax[3,0].set(ylabel="Chromagram")

for i in range(4):
    ax[i,0].label_outer()
fig.suptitle("Example Features for each class (before averaging over time)")
plt.show()  


# TRAINING
for feature_name, feature_data in features.items():
    X_train, X_test, y_train, y_test = train_test_split(feature_data, labels, test_size=0.05, random_state=1) # random_state for to ensure same split for all tests

    
    print(f"\nFitting classifier on '{feature_name}' features...")
    classifier, train_error, val_error = train_classifier(X_train, y_train) # generate and train new classifier on current features
    print(f"{train_error=}")
    print(f"{val_error=}")

    test_error = 1 - classifier.score(X_test, y_test)
    print(f"{test_error=}")
