from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
import librosa
import os
import wave
import glob
import re
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
    # record emotions to be predicted 
    emotions = emotions=args.emotions.split(",")

    # if form of output is excel
    if(sys.argv[3] == "excel"):

        # initialise workbook 
        wb = load_workbook('predict_from_audio/prediction.xlsx')
        wb.remove(wb['predictions'])
        wb.create_sheet('predictions')
        sheet = wb['predictions']
        sheet["A1"] = "emotion to predict"
        sheet["B1"] = "result"

        rows = 2
        cols = 1

        # for emotions to be predicted 
        for i in range(len(emotions)):

            # for the filepath containing that emotion
            for filepath in glob.iglob("predict_from_audio" + os.sep + "emotion testing audio 16k" + os.sep + emotions[i] + os.sep + "/*"):
                # record the emotion in the excel sheet
                sheet[get_column_letter(i + 3) + "1"] =  emotions[i]

                # record emotion to be predicted and if the prediction was correct

                duration.append(librosa.get_duration(filename = filepath))

                # record prediction probability
                start_predict = time.perf_counter()
                predictions = detector.predict_proba(filepath)
                end_predict = time.perf_counter()


                if(emotions[i]==max(predictions, key=predictions.get).lower()):
                    sheet[get_column_letter(cols) + str(rows)] = str(emotions[i])
                    sheet[get_column_letter(cols + 1) + str(rows)] = "correct"
                    cols += 2        
                else:
                    sheet[get_column_letter(cols) + str(rows)] = str(emotions[i])
                    sheet[get_column_letter(cols + 1) + str(rows)] = "incorrect"
                    cols += 2

                time_taken.append(end_predict - start_predict)

                for value in (predictions).values():
                    sheet[get_column_letter(cols) + str(rows)] = value
                    cols += 1
                rows += 1
                cols = 1

        wb.save('predict_from_audio/prediction.xlsx')
    
    else:
        # open file to write predictions on 
        with open(file = 'predict_from_audio' + os.sep + 'predictions.txt', mode  = 'w') as file:

            # iterate through all the files
            file.write("results," + str(emotions) + "\n")
            for emotion in emotions:

                # record emotions to predict to write to the putput file later
                to_predict = emotion + (8-len(emotions))*(" ")

                for filepath in glob.iglob("predict_from_audio" + os.sep + "emotion testing audio 44k" + os.sep + emotions + os.sep + "/*"):
                    
                    # write if prediction was correct
                    if(emotions==detector.predict(filepath).lower()):
                        file.write(to_predict + " correct  :" )
                    else:
                        file.write(to_predict + " incorrect:" )
                    
                    # Write probabiltiy distribution
                    for value in (detector.predict_proba(filepath)).values():
                        file.write(str(value) + ",")
                    file.write(str(filepath) + "\n")
    




