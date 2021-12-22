from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
import librosa
import os
import glob
from sys import byteorder
import sys
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

from utils import get_best_estimators

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    print(estimators_str)
    import argparse
    parser = argparse.ArgumentParser(description="""
                                    Testing emotion recognition system using your voice, 
                                    please consider changing the model and/or parameters as you wish.
                                    """)
    parser.add_argument("-e", "--emotions", help=
                                            """Emotions to recognize separated by a comma ',', available emotions are
                                            "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise)
                                            and "boredom", default is "sad,neutral,happy"
                                            """, default=sys.argv[1] )
    parser.add_argument("-m", "--model", help=
                                        """
                                        The model to use, 8 models available are: "SVC","AdaBo
                                        ostClassifier","RandomForestClassifier","GradientBoost
                                        ingClassifier","DecisionTreeClassifier","KNeighborsCla
                                        ssifier","MLPClassifier","BaggingClassifier", default
                                        is "BaggingClassifier"
                                        """.format(estimators_str), default=sys.argv[2])

    parser.add_argument("--tess_ravdess", default = True)       # use tess/ravdess dataset
    parser.add_argument("--classification", default = True)     # use classification
    parser.add_argument("--custom_db", default = True)          # use custom dataset
    parser.add_argument("--emodb", default = True)              # use emodb
    parser.add_argument('--model_name', default = os.path.join(sys.argv[4],sys.argv[2]))

    

    # Parse the arguments passed
    args, unknown = parser.parse_known_args()

    features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]


    time_taken = []
    duration = []

      # Random Forest, Adaboost  Classifier not working

       # if classifier is SVC need to parse probability as true to display probability
    if(sys.argv[2] == "SVC"):
        detector = EmotionRecognizer(model = SVC(probability = True) , emotions=args.emotions.split(","), model_name = args.model_name,  features=features , verbose=0)

    # similar for decision tree classifier 
    elif(sys.argv[2] == "DecisionTreeClassifier"):
        detector = EmotionRecognizer(model = DecisionTreeClassifier() , emotions=args.emotions.split(","), model_name = args.model_name, features=features, verbose=0)
    
    elif(sys.argv[2] == "RNN"):
        detector = DeepEmotionRecognizer(emotions=(sys.argv[1]).split(","), emodb = True, customdb = True, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)

    else:
        detector = EmotionRecognizer(estimator_dict[args.model] , emotions=args.emotions.split(","), model_name = args.model_name, features=features, verbose=0)


    # for the filepath containing that emotion
    emotions = ["neutral","calm","happy","sad","angry","fear",'disgust','ps','boredom']
    #predictions = detector.predict_proba('01_01_01_02_dogs-sitting_happy.wav')

    for emotion in emotions:
        for filepath in glob.iglob(os.path.join("predict_from_audio", f"emotion testing audio {sys.argv[4]}", f"{emotion}/*")):

            # record emotion to be predicted and if the prediction was correct
            duration.append(librosa.get_duration(filename = filepath))

            # record prediction probability
            start_predict = time.perf_counter()
            predictions = detector.predict_proba(filepath)
            end_predict = time.perf_counter() 

            time_taken.append(end_predict - start_predict)

    
  
    median_time = statistics.median(time_taken)
    median_length = statistics.median(duration)

    gs = gridspec.GridSpec(2, 2)

    bars = 30
    samples = list(range(1,42))

    fig = plt.figure()
    

    ax1 = fig.add_subplot(gs[0, 0])
    y, x, _ = ax1.hist(x = time_taken, bins = bars)
    ax1.plot([median_time, median_time],[0, y.max()], label = f"median: {round(median_time,2)} seconds")
    ax1.set_xlabel("Time taken to predict (s)")
    ax1.set_ylabel("frequency")
    ax1.legend(loc="upper right")

    ax2 = fig.add_subplot(gs[0, 1])
    y,x,_ = ax2.hist(x = duration, bins = bars)
    ax2.plot([median_length,median_length],[0, y.max()], label = f"median: {round(median_length,2)} seconds")
    ax2.set_xlabel("Length of audio (s)")
    ax2.set_ylabel("frequency")
    ax2.legend(loc="upper right")

    ax3 = fig.add_subplot(gs[1, :])
    ax3.plot(samples, time_taken, label = "Time taken to predict audio")
    ax3.plot(samples, duration, label = "length of audio")
    ax3.set_xlabel("Audio samples")
    ax3.set_ylabel("time (seconds)")
    ax3.legend(loc="upper right")

    fig.suptitle(f"Time taken to predict audio at {sys.argv[4]} using {sys.argv[2]}")
    
    plt.tight_layout()
    plt.show()
