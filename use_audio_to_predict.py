from emotion_recognition import EmotionRecognizer
from deep_emotion_recognition import DeepEmotionRecognizer
import json
import os
import glob
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
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter

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

            # create detector instance
            if(model == "DecisionTreeClassifier"):
                detector = EmotionRecognizer(model = DecisionTreeClassifier() , emotions=emotions, model_name = model_name, features=features, verbose=0)
            else:
                detector = EmotionRecognizer(estimator_dict[model] , emotions=emotions, model_name = model_name, features=features, verbose=0)

            # if predicting a single audio
            if(test_mode == 'single'):

                # load settings and record length odf audio
                single_settings = test_settings["Testing single"][0]
                audio = single_settings["Audio directory"]
                print(f'\nChosen to test a single audio using {model} trained on {model_ver}')
                print(f'Length of audio: {librosa.get_duration(filename = audio)} seconds')

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
                print(f"\nTime it took to predict: {(end_predict - start_predict)*1000} ms")
        
            # if predicting multiple audio
            else:
                multiple_settings = test_settings["Testing multiple"][0]
                output = multiple_settings["output"]

                if(output == "excel"):
                    # initialise workbook 
                    wb = load_workbook('predict_from_audio/prediction.xlsx')
                    wb.remove(wb['predictions'])
                    wb.create_sheet('predictions')
                    sheet = wb['predictions']
                    sheet["A1"] = "emotion to predict"
                    sheet["B1"] = "result"

                    rows = 2
                    cols = 1

                    sheet[get_column_letter(cols) + str(rows)] = "sm audio"
                    rows += 1

                    # Predict SM audio 
                    for i in range(len(emotions)):

                        # for the filepath containing that emotion
                        for filepath in glob.iglob("predict_from_audio" + os.sep + f"emotion testing audio {model_ver[:3]}" + os.sep + emotions[i] + os.sep + "/*"):
                            # record the emotion in the excel sheet
                            sheet[get_column_letter(i + 3) + "1"] =  emotions[i]

                            # calculate probability
                            predictions = detector.predict_proba(filepath)
                            sheet[get_column_letter(cols) + str(rows)] = str(emotions[i])

                            # record if the perediction is correct
                            if(emotions[i]==max(predictions, key=predictions.get).lower()):
                                sheet[get_column_letter(cols + 1) + str(rows)] = "correct"
                                cols += 2        
                            else:
                                sheet[get_column_letter(cols + 1) + str(rows)] = f"incorrect {max(predictions, key=predictions.get).lower()}"
                                cols += 2

                            # record the result
                            for value in (predictions).values():
                                sheet[get_column_letter(cols) + str(rows)] = value
                                cols += 1

                            rows += 1
                            cols = 1
                    
                    sheet[get_column_letter(cols) + str(rows)] = "Nene audio"
                    rows += 1

                    # Predict Nene audio
                    for audio in os.listdir(f'predict_from_audio/Nene_{model_ver[:3]}'):
                        sentiment, _ = audio.split("_")

                        # Get prediction and record the correct sentiment
                        predictions = detector.predict_proba(f'predict_from_audio/Nene_{model_ver[:3]}/{audio}')
                        sheet[get_column_letter(cols) + str(rows)] = str(sentiment)

                        # record if the perediction is correct
                        if(sentiment==max(predictions, key=predictions.get).lower()):
                            sheet[get_column_letter(cols + 1) + str(rows)] = "correct"
                            cols += 2        
                        else:
                            sheet[get_column_letter(cols + 1) + str(rows)] = f"incorrect {max(predictions, key=predictions.get).lower()}"
                            cols += 2
  
                        # record the result
                        for value in (predictions).values():
                            sheet[get_column_letter(cols) + str(rows)] = value
                            cols += 1

                        rows += 1
                        cols = 1
                    
                    # predict JL corpus audio
                    sheet[get_column_letter(cols) + str(rows)] = "JLCorpus audio"
                    rows += 1
                    
                    for audio in os.listdir(f'predict_from_audio/JL_{model_ver[:3]}'):
                        gender, sentiment, _1, _2 = audio.split("_")
                        # predict
                        predictions = detector.predict_proba(f'predict_from_audio/JL_{model_ver[:3]}/{audio}')
                        sheet[get_column_letter(cols) + str(rows)] = str(sentiment)

                        # check if prediction is correcyt
                        if(sentiment==max(predictions, key=predictions.get).lower()):
                            sheet[get_column_letter(cols + 1) + str(rows)] = "correct"
                            cols += 2        
                        else:
                            sheet[get_column_letter(cols + 1) + str(rows)] = f"incorrect {max(predictions, key=predictions.get).lower()}"
                            cols += 2

                        # record prediction
                        for value in (predictions).values():
                            sheet[get_column_letter(cols) + str(rows)] = value
                            cols += 1

                        rows += 1
                        cols = 1

                    wb.save('predict_from_audio/prediction.xlsx')
                    print('predictions saved to predict_from_audio/prediction.xlsx')
        
                # else if outputting to text
                else:
                    # open file to write predictions on 
                    with open(file = 'predict_from_audio' + os.sep + 'predictions.txt', mode  = 'w') as file:

                        # iterate through all the files
                        file.write("results," + str(emotions) + "\n")
                        for emotion in emotions:

                            # record emotions to predict to write to the putput file later
                            to_predict = emotion + (8-len(emotion))*(" ")

                            for filepath in glob.iglob(os.path.join("predict_from_audio",f"emotion testing audio {model_ver[:3]}", f"{emotion}/*")):
                                
                                # write if prediction was correct
                                if(emotion == detector.predict(filepath).lower()):
                                    file.write(to_predict + " correct   :" )
                                else:
                                    file.write(to_predict + " incorrect :" )
                                
                                # Write probabiltiy distribution
                                for value in (detector.predict_proba(filepath)).values():
                                    file.write(str(value) + ",")
                                file.write(str(filepath) + "\n")
                            
                    print('predictions saved to predict_from_audio/prediction.txt')
        else:
            print('Training not implemented here yet')
    


                

    