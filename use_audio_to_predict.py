from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
import pyaudio
import os
import wave
from sys import byteorder
import sys
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import time

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

    parser.add_argument("-t","--tess_ravdess", help= True)      # using tess and ravdess to train
    parser.add_argument("-c","--classification", help= True)    # using classification method
    parser.add_argument('--model_name', default = sys.argv[2])


    # Parse the arguments passed
    args, unknown = parser.parse_known_args()

    features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
    
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

    # train the model and print the confusion matrix

    #predict from filename passed in args
    start_predict = time.perf_counter()
    result = detector.predict_proba(sys.argv[3])
    end_predict = time.perf_counter()
    print(result)
    print(max(result, key=result.get))

    print("Time it took to predict:", end_predict - start_predict)
    