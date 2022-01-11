from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
import json
import os
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
import librosa

from utils import get_best_estimators

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

if __name__ == "__main__":
    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    print(estimators_str)

    with open('predict.json', 'r') as config_file:
        data = json.load(config_file)
        mandatory_settings =    data["Mandatory Settings"][0]
        Test_or_train_mode =    mandatory_settings["Test or Train"].lower()

        # check if train or test, if not then exit
        if((Test_or_train_mode!= "train") and (Test_or_train_mode != "test")) :
            sys.exit("Please choose whether to Test or Train.\n This can be done under Mandatory Settings in predict.json")       
    
        # load mandatory settings
        model =     mandatory_settings["model"].format(estimators_str)
        model_ver = mandatory_settings["model_ver"]
        emotions =  mandatory_settings['emotions'].split(",")
        features =  mandatory_settings["features"].split(",")
        model_name = os.path.join(model_ver,model)

        # if testing
        if(Test_or_train_mode == "test"):

            # load testing settings
            test_settings = data["Testing settings"][0]
            test_mode =     test_settings["Test mode"].lower()

            # if predicting a single audio
            if(test_mode == 'single'):

                # load settings and record length odf audio
                single_settings = test_settings["Testing single"][0]
                audio = single_settings["Audio directory"]
                print(f'\nLength of audio recorded: {librosa.get_duration(filename = audio)} seconds')

                # create detector instance
                if(model == "SVC"):
                    detector = EmotionRecognizer(model = estimator_dict[model] , emotions=emotions, model_name = model_name,  features=features , verbose=0)
                else:
                    detector = EmotionRecognizer(estimator_dict[model] , emotions=emotions, model_name = model_name, features=features, verbose=0)

                #predict from filename passed in args
                start_predict = time.perf_counter()
                result = detector.predict_proba(audio)
                end_predict = time.perf_counter()

                # print result
                print(f'\n{result}')
                maximum = max(result, key=result.get)
                max_value = result[maximum]
                del result[maximum]

                second = max(result, key=result.get)
                second_value = result[second]

                print(f"\nfirst prediction  : {maximum} \nsecond prediction : {second} \ndifference is {(max_value - second_value)*100} %")
                print(f"\nTime it took to predict: {end_predict - start_predict} s")
        
            else:
                multiple_settings = test_settings["Testing multiple"][0]
                output = multiple_settings["output"][0]
        
        else:
            print('Training not implemented here yet')
    


                

    