from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
from sys import byteorder
import json
import os
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import argparse
import time

from utils import get_best_estimators

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

if __name__ == "__main__":

    # iterate through all models

    start_train = time.perf_counter()

    models = ["KNeighborsClassifier","SVC","GradientBoostingClassifier","DecisionTreeClassifier","MLPClassifier","BaggingClassifier"]

    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)

    with open('predict.json') as config_file:
        data = json.load(config_file)
        model_ver = data["model_ver"]
        emotions =  data['emotions'].split(",")
        features =  data["features"].split(",")

    #models = ["RandomForestClassifier"]
    for model in models:
        model_name = os.path.join(model_ver,model)
        
       
        # Random Forest, Adaboost  Classifier not working so display models that fail to train

        # try:
        #     # if classifier is SVC need to parse probability as true to display probability
        #     if(model == "SVC"):
        #         detector = EmotionRecognizer(model = SVC(probability = True),emotions=emotions, model_name = model_name, features=features, verbose=0)

        #     # similar for decision tree classifier 
        #     elif(model == "DecisionTreeClassifier"):
        #         detector = EmotionRecognizer(model = DecisionTreeClassifier() , emotions=emotions, model_name = model_name, features=features, verbose=0)
            
        #     elif(model == "RNN"):
        #         detector = DeepEmotionRecognizer(emotions=(sys.argv[1]).split(","), model_name = model_name,  emodb = True, customdb = True, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)

        #     else:
        #         detector = EmotionRecognizer(estimator_dict[args.model] , emotions=emotions, model_name = model_name, features=features, verbose=0)
            
        #     # train the model and display status
        #     detector.train()
        #     print(f"\n{model} trained")
        #     print(detector.confusion_matrix())
        #     print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))
            
        # except Exception as e:
        #     print(f"{model} failed to train")
        #     continue

        if(model == "SVC"):
            detector = EmotionRecognizer(model = SVC(probability = True),emotions=emotions, model_name = model_name, features=features, verbose=0)

        # similar for decision tree classifier 
        elif(model == "DecisionTreeClassifier"):
            detector = EmotionRecognizer(model = DecisionTreeClassifier() , emotions=emotions, model_name = model_name, features=features, verbose=0)
        
        elif(model == "RNN"):
            detector = DeepEmotionRecognizer(emotions=emotions, model_name = model_name,  emodb = True, customdb = True, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)

        else:
            detector = EmotionRecognizer(estimator_dict[model] , emotions=emotions, model_name = model_name, features=features, verbose=0)
        
        # train the model and display status
        detector.train()
        print(f"\n{model} trained")
        print(detector.confusion_matrix())
        print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))

    end_train = time.perf_counter()
    print(f"\nThis process took {end_train - start_train} seconds")
        