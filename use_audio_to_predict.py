from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
import json
import os
from sys import byteorder
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

    with open('predict.json') as config_file:
        data = json.load(config_file)
        model =     data["model"].format(estimators_str)
        model_ver = data["model_ver"]
        emotions =  data['emotions'].split(",")
        audio =     data["audio"]
        features =  data["features"].split(",")
    
    model_name = os.path.join(model_ver,model)
    
    # Random Forest, Adaboost  Classifier not working
    # if classifier is SVC need to parse probability as true to display probability
    if(model == "SVC"):
        detector = EmotionRecognizer(model = estimator_dict[model] , emotions=emotions, model_name = model_name,  features=features , verbose=0)
    
    elif(model == "RNN"):
        detector = DeepEmotionRecognizer(emotions=emotions, emodb = True, customdb = True, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)

    else:
        detector = EmotionRecognizer(estimator_dict[model] , emotions=emotions, model_name = model_name, features=features, verbose=0)

    # train the model and print the confusion matrix
    result = detector.predict_proba("01_01_01_02_dogs-sitting_happy.wav")

    #predict from filename passed in args
    start_predict = time.perf_counter()
    result = detector.predict_proba(audio)
    end_predict = time.perf_counter()

    print(result)
    maximum = max(result, key=result.get)
    max_value = result[maximum]
    del result[maximum]

    second = max(result, key=result.get)
    second_value = result[second]

    print(f"\nfirst prediction  : {maximum} \nsecond prediction : {second} \ndifference is {(max_value - second_value)*100} %")

    print(f"\nTime it took to predict: {end_predict - start_predict} s")
    