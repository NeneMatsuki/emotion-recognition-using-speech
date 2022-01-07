from convert_wavs import convert_audio
from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
import librosa
import os
import glob
import numpy as np
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import statistics
import json
import pandas as pd
from scipy.stats import gaussian_kde
import seaborn as sns
import soundfile as sf

from utils import get_best_estimators

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

if __name__ == "__main__":
    # gets tuned parameters from gridsearch 
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)

    # get details from predict.json file
    with open('predict.json') as config_file:
        data = json.load(config_file)
        model =     data["model"].format(estimators_str)
        emotions =  data['emotions'].split(",")
        frequency = data["frequency"]
        features =  data["features"].split(",")
        model_ver = data["model_ver"]
    
    model_name = os.path.join(model_ver,model)

    # initialise list to start time taken and 
    time_taken = []
    duration = []

    # if classifier is SVC need to parse probability as true to display probability
    if(model == "SVC"):
        detector = EmotionRecognizer(model = SVC(probability = True) , emotions=emotions, model_name = model_name,  features=features , verbose=0)

    # similar for decision tree classifier 
    elif(model == "DecisionTreeClassifier"):
        detector = EmotionRecognizer(model = DecisionTreeClassifier() , emotions=emotions, model_name = model_name, features=features, verbose=0)
    
    elif(model == "RNN"):
        detector = DeepEmotionRecognizer(emotions=emotions, emodb = True, customdb = True, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)

    else:
        detector = EmotionRecognizer(estimator_dict[model] , emotions=emotions, model_name = model_name, features=features, verbose=0)

    # for the filepath containing that emotion
    emotions = ["neutral","calm","happy","sad","angry","fear",'disgust','ps','boredom']

    for emotion in emotions:
        for filepath in glob.iglob(os.path.join("predict_from_audio", f"emotion testing audio {frequency}", f"{emotion}/*")):

            # record emotion to be predicted and if the prediction was correct
            duration.append(librosa.get_duration(filename = filepath))

            # record prediction probability and time
            if(model_ver[:3] != frequency):
                start_predict = time.perf_counter()
                y, s = librosa.load(filepath, sr=44100)
                sf.write("temp.wav", y, s)
                end_predict = time.perf_counter() 


            else:
                start_predict = time.perf_counter()
                predictions = detector.predict_proba(filepath)
                end_predict = time.perf_counter() 

            time_taken.append(end_predict - start_predict)

    # for filepath in os.listdir(os.path.join('data','training')):

    #     for audio in os.listdir(os.path.join('data','training',filepath)):

    #         # record emotion to be predicted and if the prediction was correct
    #         audio = os.path.join('data','training',filepath,audio)
    #         duration.append(librosa.get_duration(filename = audio))

    #         # record prediction probability and time
    #         start_predict = time.perf_counter()
    #         predictions = detector.predict_proba(audio)
    #         end_predict = time.perf_counter() 

    #         time_taken.append(end_predict - start_predict)

    # record medians
    median_time = statistics.median(time_taken)
    mean_time = statistics.mean(time_taken)

    median_length = statistics.median(duration)
    mean_length = statistics.mean(duration)

    # sort sample length and in ascending order of audio length
    index = np.argsort(duration)
    sorted_duration = np.zeros(len(duration))
    sorted_time_taken = np.zeros(len(index))

    for i in range(len(index)):
        sorted_time_taken[i] = time_taken[index[i]]
        sorted_duration[i] = duration[index[i]]

    # create subplot axes to plot on 
    gs = dict(width_ratios=[1, 1], height_ratios=[1, 1, 2])
    fig, axd = plt.subplot_mosaic([['upper left', 'right'],
                               ['mid left', 'right'],
                               ['lower','lower']],
                              gridspec_kw=gs,
                              constrained_layout=True)
    
    # plot histogram and probability density of the time taken to predict
    ax1  = axd['upper left']
    sns.distplot(a = time_taken, bins = 20, ax = ax1)

    ax1.axvline(x = median_time, color = 'y' , label = f"median: {round(median_time,2)} seconds")
    ax1.axvline(x = mean_time, color = 'm', label = f"mean: {round(mean_time,2)} seconds")
    ax1.set_xlabel("Time taken to predict (s)")
    ax1.set_ylabel("Probability density")
    ax1.legend(loc="upper right")

    # plot histogram and probability density for the duration of audio
    ax2 = axd['mid left']
    sns.distplot(x = duration, bins = 20, ax = ax2)

    ax2.axvline(x = median_length, color = 'y', label = f"median: {round(median_length,2)} seconds")
    ax2.axvline(x = mean_length, color = 'm', label = f"mean: {round(mean_length,2)} seconds")
    ax2.set_xlabel("Length of audio (s)")
    ax2.set_ylabel("Probability Density")
    ax2.legend(loc="upper right")

    # plot time it took to predict and the length of the audio together, in ascending order of the length of audio
    ax3 = axd['lower']
    ax4 = ax3.twinx()
    lns1 = ax3.plot(list(range(1,len(time_taken)+1)), sorted_time_taken, '-b.', label = "Time taken to predict audio")
    lns2 = ax4.plot(list(range(1,len(time_taken)+1)), sorted_duration, '-r.', label = "length of audio")

    ax4.set_ylabel("length of audio (seconds)")
    ax3.set_xlabel("Audio in ascending order by length")
    ax3.set_ylabel("time taken to predicr(seconds)")
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc="upper left")

    # plot a scatter plot of the time taken to predict and the length of the corresponding audio
    ax5 = axd['right']
    ax5.plot(time_taken, duration, 'r.')
    ax5.set_xlabel('time taken to predict audio (s)')
    ax5.set_ylabel('length of audio (s)')

    # add a title and lplot
    fig.suptitle(f"Time taken to predict audio at {frequency} using {model}")
    plt.tight_layout()
    plt.show()
