
# Edited [emotion recognition using speech](https://github.com/x4nth055/emotion-recognition-using-speech) to make a Audio sentiment analysis model

Please read [The original README](https://github.com/NeneMatsuki/emotion-recognition-using-speech/blob/master/README_original.md) for more information

## Introduction
- This repository handles building and training Audio Emotion Recognition System.
- The idea behind this tool is to create an interface to train/test and save a suited machine learning algorithm that could recognize and detect human emotions from speech.

## Requirements
- **Python 3.8**
### Python Packages
- librosa==0.8.1
- numpy==1.18.5
- pandas==1.3.4
- soundfile==0.10.3.post1
- wave==0.0.2
- sklearn==0.0
- tqdm==4.62.3
- matplotlib==3.5.0
- tensorflow==2.3.0
- pyaudio==0.2.11
- scikit-learn==1.0.1
- seaborn==0.11.2
- openpyxl==3.0.9 (optional) : used if preffered method of output after predicting multiple audio is excel
- [ffmpeg](https://ffmpeg.org/) (optional): used if you want to add more sample audio by converting to 16000Hz sample rate and mono channel which is provided in ``convert_wavs.py``

Install these libraries and pyaudio by the following command:
```
pip install pipwin
pipwin install pyaudio
pip install -r requirements.txt
```

### Dataset
This repository used 5 datasets (including this repo's custom dataset) which are downloaded and formatted already in `data` folder:
- [**RAVDESS**](https://zenodo.org/record/1188976) : The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.
- [**TESS**](https://tspace.library.utoronto.ca/handle/1807/24487) : **T**oronto **E**motional **S**peech **S**et that was modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966). A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years).
- [**EMO-DB**](http://emodb.bilderbar.info/docu/) : As a part of the DFG funded research project SE462/3-1 in 1997 and 1999 we recorded a database of emotional utterances spoken by actors. The recordings took place in the anechoic chamber of the Technical University Berlin, department of Technical Acoustics. Director of the project was Prof. Dr. W. Sendlmeier, Technical University of Berlin, Institute of Speech and Communication, department of communication science. Members of the project were mainly Felix Burkhardt, Miriam Kienast, Astrid Paeschke and Benjamin Weiss.
- [""JL-Corpus**](https://github.com/tli725/JL-Corpus) : Emotional speech corpus in New Zealand English (Jesin James, Li Tian, Catherine Watson, "An Open Source Emotional Speech Corpus for Human Robot Interaction Applications", in Proc. Interspeech, 2018)
- **Custom** : SM audio from voice actors, happy, angry, neutral, sad. Please add more audio files here if needed, formatted in the form audio_emotion.wav.



### Emotions available
There are 9 emotions available: "neutral", "calm", "happy" "sad", "angry", "fear", "disgust", "ps" (pleasant surprise) and "boredom".
## Feature Extraction
Feature extraction is the main part of the speech emotion recognition system. It is basically accomplished by changing the speech waveform to a form of parametric representation at a relatively lesser data rate.

In this repository, we have used the most used features that are available in [librosa](https://github.com/librosa/librosa) library including:
- [MFCC](https://en.wikipedia.org/wiki/Mel-frequency_cepstrum)
- Chromagram 
- MEL Spectrogram Frequency (mel)
- Contrast
- Tonnetz (tonal centroid features)

### Classifiers
- SVC
- GradientBoostingClassifier
- KNeighborsClassifier
- MLPClassifier
- BaggingClassifier

# Testing/Training
Please configure `test_train_input_config.json` by following the introductions in it and reun ` audio_emotion_recognizer.py` or type into the terminal: `python audio_emotinon_recognizer.py`. 

The recommended model to choose in this config file is the **MLP Classifier trained on 16000 Hz audio using 3 features  detecting angry, happy, sad and neutral** which has approximately **96% accuracy**.  Choosing this model in the config file looks like this:
```.json
"MANDATORY FIELD SETTING": {

	"Model name" :"MLPClassifier",
	"Pre-saved model folder" :"16k_3feat_JL",
	"Emotions" :"angry,happy,neutral,sad",
	"Features" :"mfcc,chroma,mel",
	.....
``` 
In later fields,  choose to:
### 1. **Test a single audio from an audio directory, or live from your voice. This outputs to command line:**
```   
    emotion probabilities
    {'angry': 1.451255905740051e-09, 'happy': 0.00016189376136972053, 'neutral': 0.9989209465295328, 'sad': 0.0009171582578415177}

    Best prediction  : neutral
    second best prediction : sad
    difference is 99.80037882716913 %

    Therefore predicted neutral with an effective probability of 99.80037882716913 %
    Time it took to predict: 81.84999999999931 ms
```
### 2. **Test multiple audio files (with all your testing data or a subset). This gives:**
- Predictions recorded in an [excel sheet](test_audio/prediction.xlsx) or in a [text file](test_audio/prediction.rxr) (choose in  `test_train_input_config.json`)

- Model performance (inference time) as a visualization:
	 (can be enabled/disabled in `test_train_input_config.json`
	 
	![16k_3feat_JL_MLPClassifier_all](https://user-images.githubusercontent.com/80789350/151467119-18682be6-2e50-4e73-b99c-441504d22b6b.png)*Please note that depending on what audio files you put in the folder [data](data)*

### 3. **Train single or multiple model(s), this will show the confusion matrix of the model(s) trained**
```
MLPClassifier trained
Number of samples: 9896
              predicted_angry  predicted_happy  predicted_neutral  predicted_sad
true_angry          95.634598         2.384802           0.161681       1.818917
true_happy           2.546484        94.704933           0.161681       2.586904
true_neutral         0.000000         0.161681          99.191597       0.646726
true_sad             1.010509         2.384802           0.485044      96.119644
Test accuracy score: 96.413%
```


