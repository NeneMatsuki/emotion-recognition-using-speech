from deep_emotion_recognition import DeepEmotionRecognizer
import sys
import glob
import os
import argparse
# initialize instance
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)

# train the model
parser = argparse.ArgumentParser()
parser.add_argument("--tess_ravdess", default = True)
parser.add_argument("--classification", default = True)
parser.add_argument("--custome_db", default = True)
parser.add_argument("--emodb", default = False)

args, unknown = parser.parse_known_args()
deeprec = DeepEmotionRecognizer(emotions=(sys.argv[1]).split(","), emodb = False, customdb = True, n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)

deeprec.train()

# get the accuracy
print(f"Prediction accuracy: {deeprec.test_score()}")


with open(file = 'predict_from_audio' + os.sep + 'predictions.txt', mode  = 'w') as file:

    emotions = sys.argv[1].split(",")
    
    for i in range(len(emotions)):
        for filepath in glob.iglob("predict_from_audio" + os.sep + "emotion testing audio 44k" + os.sep + emotions[i] + os.sep + "/*"):
        # write probabilities in output
            file.write(str(filepath) + ",")
            for value in (deeprec.predict_proba(filepath)).values():
                file.write(str(value) + ",")
            file.write('\n')     
