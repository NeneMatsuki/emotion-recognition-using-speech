from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
import librosa
import os
import glob
from array import array
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

from utils import get_best_estimators

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    print(estimators_str)

    with open('predict_single.json') as config_file:
        data = json.load(config_file)
        model =     data["model"].format(estimators_str)
        emotions =  data['emotions'].split(",")
        frequency = data["frequency"]
        features =  data["features"].split(",")
        model_ver = data["model_ver"]
    
    model_name = os.path.join(model_ver,model)

    # initialise array to start time talen and 
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
        for filepath in glob.iglob(os.path.join("predict_from_audio", f"emotion testing audio 16k", f"{emotion}/*")):

            # record emotion to be predicted and if the prediction was correct
            duration.append(librosa.get_duration(filename = filepath))

            # record prediction probability and time
            start_predict = time.perf_counter()
            predictions = detector.predict_proba(filepath)
            end_predict = time.perf_counter() 

            time_taken.append(end_predict - start_predict)

    # record medfians
    median_time = statistics.median(time_taken)
    median_length = statistics.median(duration)

    gs = gridspec.GridSpec(2, 2)

    bars = 30
    samples = list(range(1,42))

    fig = plt.figure()
    
    # plot for time taken to predict 
    ax1 = fig.add_subplot(gs[0, 0])
    y, x, _ = ax1.hist(x = time_taken, bins = bars)
    ax1.plot([median_time, median_time],[0, y.max()], label = f"median: {round(median_time,2)} seconds")
    ax1.set_xlabel("Time taken to predict (s)")
    ax1.set_ylabel("frequency")
    ax1.legend(loc="upper right")

    # plot for duration of audio
    ax2 = fig.add_subplot(gs[0, 1])
    y,x,_ = ax2.hist(x = duration, bins = bars)
    ax2.plot([median_length,median_length],[0, y.max()], label = f"median: {round(median_length,2)} seconds")
    ax2.set_xlabel("Length of audio (s)")
    ax2.set_ylabel("frequency")
    ax2.legend(loc="upper right")

    # plot for each audio and the length it tool to predict 
    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(samples, time_taken, label = "Time taken to predict audio")
    ax3.plot(samples, duration, label = "length of audio")
    ax3.set_xlabel("Audio samples")
    ax3.set_ylabel("time (seconds)")
    ax3.legend(loc="upper right")

    fig.suptitle(f"Time taken to predict audio at {frequency} using {model}")
    
    plt.tight_layout()
    plt.show()
