import os
from os.path import dirname, join
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import numpy as np
import scipy.io.wavfile as wavfile
from matplotlib import pyplot as plt
train_split = 0.8
data_dir = join(dirname(__file__), 'data')

clsnames = ['gel', 'pia', 'voi', 'cel']
sample_rate = 44100
data = []
labels = []

for label, clsname in enumerate(clsnames):
    clsdir = join(data_dir, clsname)
    for file in os.listdir(clsdir):
        if file.endswith('.wav'):
            _, stereo_data = wavfile.read(join(clsdir, file))            
            data.append(stereo_data)
            labels.append(label)

data = np.array(data)
data = data[:,:,0] # drop second channel
labels = np.array(labels)

ts = np.linspace(0, data.shape[1] / sample_rate, data.shape[1])
plt.plot(ts, data[0])
plt.show()


# pipeline = Pipeline([
#     ('clf', RandomForestClassifier(n_estimators=250)),
# ])

# def InstrumentClassifier(train_data, train_labels, test_data):
#     clf_final = RandomForestClassifier(n_estimators=250)
#     clf_final.fit(train_data, train_labels)
#     labels = clf_final.predict(test_data)
#     return labels

