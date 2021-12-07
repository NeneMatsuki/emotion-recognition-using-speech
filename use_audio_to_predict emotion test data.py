from emotion_recognition import EmotionRecognizer
import pyaudio
import os
import wave
import glob
import re
from sys import byteorder
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
                                            """, default="neutral,calm,happy,sad,angry,fear,disgust,ps,boredom" )
    parser.add_argument("-m", "--model", help=
                                        """
                                        The model to use, 8 models available are: "SVC","AdaBo
                                        ostClassifier","RandomForestClassifier","GradientBoost
                                        ingClassifier","DecisionTreeClassifier","KNeighborsCla
                                        ssifier","MLPClassifier","BaggingClassifier", default
                                        is "BaggingClassifier"
                                        """.format(estimators_str), default="KNeighborsClassifier")


    # Parse the arguments passed
    args = parser.parse_args()

    features = ["mfcc", "chroma", "mel"]
    detector = EmotionRecognizer(estimator_dict[args.model] , emotions=args.emotions.split(","), features=features, verbose=0)
    
    # for SVC
    #detector = EmotionRecognizer(model = SVC(probability = True) , emotions=args.emotions.split(","), features=features, verbose=0)
    detector.train()
    print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))


    with open(file = 'predict_from_audio' + os.sep + 'output.txt', mode  = 'w') as file:

        for filepath in glob.iglob("predict_from_audio" + os.sep + "emotion testing audio 44k/*"):
            for value in (detector.predict_proba(filepath)).values():
                file.write(str(value) + ",")
            #output = str(list((detector.predict_proba(filepath)).values()))
            file.write('\n')
 
    #print(detector.confusion_matrix(percentage=True, labeled=True))