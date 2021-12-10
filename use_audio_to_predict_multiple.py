from emotion_recognition import EmotionRecognizer
import pyaudio
import os
import wave
import glob
import re
from sys import byteorder
import sys
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

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

    parser.add_argument("--tess_ravdess", default = True)
    parser.add_argument("--classification", default = False)
    parser.add_argument("--custome_db", default = False)
    parser.add_argument("--emodb", default = False)
    parser.add_argument("--balance", default = True)
    

    # Parse the arguments passed
    args, unknown = parser.parse_known_args()

    features = ["mfcc", "chroma", "mel"]

    if(sys.argv[2] == "SVC"):
        detector = EmotionRecognizer(model = SVC(probability = True) , emotions=args.emotions.split(","), features=features, verbose=0)
    else:
        detector = EmotionRecognizer(estimator_dict[args.model] , emotions=args.emotions.split(","), features=features, verbose=0)
    

    detector.train()
    print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))

    
    # oprn file  
    with open(file = 'predict_from_audio' + os.sep + 'predictions.txt', mode  = 'w') as file:

        # iterate through all the files

        emotions = emotions=args.emotions.split(",")

        file.write("results," + str(emotions) + "\n")
        for i in range(len(emotions)):

            # record emotions to predict to write to the putput file later
            to_predict = emotions[i] + (8-len(emotions[i]))*(" ")

            for filepath in glob.iglob("predict_from_audio" + os.sep + "emotion testing audio 44k" + os.sep + emotions[i] + os.sep + "/*"):
            # write probabilities in output

                if(emotions[i]==detector.predict(filepath).lower()):
                    file.write(to_predict + " correct  :" )
                
                else:
                    file.write(to_predict + " incorrect:" )
                
                for value in (detector.predict_proba(filepath)).values():
                    file.write(str(value) + ",")
                file.write(str(filepath) + "\n")
            