from deep_emotion_recognition import DeepEmotionRecognizer
# initialize instance
# inherited from emotion_recognition.EmotionRecognizer
# default parameters (LSTM: 128x2, Dense:128x2)
deeprec = DeepEmotionRecognizer(emotions=['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'ps', 'boredom'], n_rnn_layers=2, n_dense_layers=2, rnn_units=128, dense_units=128)
# train the model
deeprec.train()

test_audio = 'test.wav'
angry_audio = 'data/validation/Actor_10/03-02-05-02-02-02-10_angry.wav'
happy_audio = "01_01_01_02_dogs-sitting_happy.wav"
sad_audio = "01_01_02_01_dogs-sitting_sad.wav"
fear_audio = '03-02-06-01-02-01-01_fear.wav'
# get the accuracy
print(f"Prediction accuracy: {deeprec.test_score()}")
# predict angry audio sample
prediction = deeprec.predict(fear_audio)
print(f"Prediction: {prediction}")
print(deeprec.predict_proba(fear_audio))
