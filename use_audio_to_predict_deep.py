from deep_emotion_recognition import DeepEmotionRecognizer
import sys
# initialize instance
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=(sys.argv[1]).split(","), n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
# train the model
deeprec.train()

# get the accuracy
print(f"Prediction accuracy: {deeprec.test_score()}")
# predict angry audio sample
prediction = deeprec.predict(sys.argv[3])
print(f"Prediction: {prediction}")
print(deeprec.predict_proba(sys.argv[3]))
