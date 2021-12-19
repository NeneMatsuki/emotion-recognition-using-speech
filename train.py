from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
from sys import byteorder
import sys
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
import argparse

from utils import get_best_estimators

def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}

if __name__ == "__main__":

    # iterate through all models

    models = ["SVC","AdaBoostClassifier","RandomForestClassifier","GradientBoostingClassifier","DecisionTreeClassifier","KNeighborsClassifier","MLPClassifier","BaggingClassifier", "RNN"]
    
    for model in models:

        estimators = get_best_estimators(True)
        estimators_str, estimator_dict = get_estimators_name(estimators)
        
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
                                            """.format(estimators_str), default=model)

        parser.add_argument("--tess_ravdess", default = True)       # use tess/ravdess dataset
        parser.add_argument("--classification", default = True)     # use classification
        parser.add_argument("--custom_db", default = True)          # use custom dataset
        parser.add_argument("--emodb", default = True)              # use emodb
        parser.add_argument("-n", '--model_name', default = model)


        # Parse the arguments passed
        args, unknown = parser.parse_known_args()

        features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
        
        # Random Forest, Adaboost  Classifier not working so display models that fail to train

        try:
            # if classifier is SVC need to parse probability as true to display probability
            if(model == "SVC"):
                detector = EmotionRecognizer(model = SVC(probability = True),emotions=args.emotions.split(","), model_name = args.model_name, features=features, verbose=0)

            # similar for decision tree classifier 
            elif(model == "DecisionTreeClassifier"):
                detector = EmotionRecognizer(model = DecisionTreeClassifier() , emotions=args.emotions.split(","), model_name = args.model_name, features=features, verbose=0)
            
            elif(model == "RNN"):
                detector = DeepEmotionRecognizer(emotions=(sys.argv[1]).split(","), emodb = True, customdb = True, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)

            else:
                detector = EmotionRecognizer(estimator_dict[args.model] , emotions=args.emotions.split(","), model_name = args.model_name, features=features, verbose=0)
            
            # train the model and display status
            detector.train()
            print(f"\n{model} trained")
            print(detector.confusion_matrix())
            print("Test accuracy score: {:.3f}%".format(detector.test_score()*100))
            
        except Exception as e:
            print(f"{model} failed to train")
            
