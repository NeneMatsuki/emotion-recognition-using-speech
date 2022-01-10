# Edited [emotion recognition using speech](https://github.com/x4nth055/emotion-recognition-using-speech) to test quality of models with pre recorded audio.

Please read [The original README](https://github.com/NeneMatsuki/emotion-recognition-using-speech/blob/master/README_original.md) on instructions on using the model and any requirements. 

Summary of model performance found in this [spreadsheet](https://docs.google.com/spreadsheets/d/1eKX86JusWnL_1YBtDadtsKyx1cQiSuedk0V_xlTiHLw/edit?usp=sharing)

## Introduction
- This repository handles building and training Speech Emotion Recognition System.
- The basic idea behind this tool is to build and train/test a suited machine learning ( as well as deep learning ) algorithm that could recognize and detects human emotions from speech.
- This is useful for many industry fields such as making product recommendations, affective computing, etc.
- Check this [tutorial](https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn) for more information.
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
- openpyxl==3.0.9 (optional) : used if preffered method of output after predicting multiple audio is excel
- [ffmpeg](https://ffmpeg.org/) (optional): used if you want to add more sample audio by converting to 16000Hz sample rate and mono channel which is provided in ``convert_wavs.py``

Install these libraries and pyaudio by the following command:
```
pip install pipwin
pipwin install pyaudio
pip install -r requirements.txt
```

### Dataset
This repository used 4 datasets (including this repo's custom dataset) which are downloaded and formatted already in `data` folder:
- [**RAVDESS**](https://zenodo.org/record/1188976) : The **R**yson **A**udio-**V**isual **D**atabase of **E**motional **S**peech and **S**ong that contains 24 actors (12 male, 12 female), vocalizing two lexically-matched statements in a neutral North American accent.
- [**TESS**](https://tspace.library.utoronto.ca/handle/1807/24487) : **T**oronto **E**motional **S**peech **S**et that was modeled on the Northwestern University Auditory Test No. 6 (NU-6; Tillman & Carhart, 1966). A set of 200 target words were spoken in the carrier phrase "Say the word _____' by two actresses (aged 26 and 64 years).
- [**EMO-DB**](http://emodb.bilderbar.info/docu/) : As a part of the DFG funded research project SE462/3-1 in 1997 and 1999 we recorded a database of emotional utterances spoken by actors. The recordings took place in the anechoic chamber of the Technical University Berlin, department of Technical Acoustics. Director of the project was Prof. Dr. W. Sendlmeier, Technical University of Berlin, Institute of Speech and Communication, department of communication science. Members of the project were mainly Felix Burkhardt, Miriam Kienast, Astrid Paeschke and Benjamin Weiss.
- **Custom** : Some unbalanced noisy dataset that is located in `data/train-custom` for training and `data/test-custom` for testing in which you can add/remove recording samples easily by converting the raw audio to 16000 sample rate, mono channel (this is provided in `create_wavs.py` script in ``convert_audio(audio_path)`` method which requires [ffmpeg](https://ffmpeg.org/) to be installed and in *PATH*) and adding the emotion to the end of audio file name separated with '_' (e.g "20190616_125714_happy.wav" will be parsed automatically as happy)


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
- RandomForestClassifier
- GradientBoostingClassifier
- KNeighborsClassifier
- MLPClassifier
- BaggingClassifier
- Recurrent Neural Networks (Keras)
### Regressors
- SVR
- RandomForestRegressor
- GradientBoostingRegressor
- KNeighborsRegressor
- MLPRegressor
- BaggingRegressor
- Recurrent Neural Networks (Keras)
-

# Base Configuration
> This description of configuring `predict.json` would be reffered to in further steps on training and testing the models.

The Base configuration file will look like this:
```.json
{
    "comment_general": "for general predictions, training, testing",
    "model"     :"",
    "model_ver" :"16k_3feat",
    "emotions"  :"angry,happy,neutral,sad",
    "features"  :"mfcc,chroma,mel",
    "frequency" :"16k",

    "comment_single": "for single predictions",
    "audio"     :"",

    "comment_multiple": "for multiple predictions",
    "output"    :""
}
```
where:

| Configuration | Description | Used in |
| --- | --- | --- |
| model_ver  | name of folder where the model is saved | 
| emotions | emotions to predict, test, and train on |
| features | audio features used to predict, test, and train on |
| frequency | audio frequency to predict, test and train on | 

This configuration would be used to test, train, and predict on audio. 

## 1. Training the models using train.py 

> When predicting with a new combination of emotions or awith any dataset is added (i.e model has not been trained yet) Please train the models first. 
> Train.py extracts features from audio in the folder data and then trains and saves the specified models. The output is a confusion matrix of the models trained

Models available that are pre-trained and saved are in the folder `models`. available models are:

- 16k_3feat : model trained on 16k using mel spectogram, mfcc, chromagram
- 44k_3feat : model trained on 16k using mel spectogram, mfcc, chromagram
- 16k_5feat : model trained on 16k using mel spectogram, mfcc, chromagram, contrast, tonnetz
- 44k_5feat : model trained on 16k using mel spectogram, mfcc, chromagram, contrast, tonnetz

You do not need to run `train.py` to use the models listed above, but you can if you wish to print the confusion matrices

To train a new model, create a new sub folder in the `models` folder and structure it like the ones that exist ( create folders within it with models to train)

Before running `train.py`, configure `predict.json` with the [base configuration](#base-configuration) , but change model_ver to the folder created if training a new model.

Example config:

```.json
{
    "comment_general": "for general predictions, training, testing",
    "model"     :"",
    "model_ver" :"16k_3feat",
    "emotions"  :"angry,happy,neutral,sad",
    "features"  :"mfcc,chroma,mel",
    "frequency" :"16k",

    "comment_single": "for single predictions",
    "audio"     :"",

    "comment_multiple": "for multiple predictions",
    "output"    :""
}
```
runnig `train.py` using this configuration gives an output in the format:


```.txt
KNeighborsClassifier trained
              predicted_angry  predicted_happy  predicted_neutral  predicted_sad
true_angry          92.956558         3.542809           0.042176       3.458456
true_happy           4.470687        92.197380           0.168705       3.163222
true_neutral         0.126529         0.126529          99.325180       0.421763
true_sad             3.838043         5.314213           0.210881      90.636864
Test accuracy score: 93.779%

SVC trained
              predicted_angry  predicted_happy  predicted_neutral  predicted_sad
true_angry          86.461411         5.567271           0.000000       7.971320
true_happy          10.080135        74.314636           0.000000      15.605229
true_neutral         0.210881         0.253058          97.511604       2.024462
true_sad             8.393083         4.765922           0.000000      86.840996
Test accuracy score: 86.282%

GradientBoostingClassifier trained
              predicted_angry  predicted_happy  predicted_neutral  predicted_sad
true_angry          96.372833         2.024462           0.210881       1.391818
true_happy           3.838043        93.968788           0.000000       2.193167
true_neutral         0.210881         0.295234          98.987770       0.506116
true_sad             1.940110         2.150991           0.168705      95.740196
Test accuracy score: 96.267%

DecisionTreeClassifier trained
              predicted_angry  predicted_happy  predicted_neutral  predicted_sad
true_angry          80.303665        10.839308           0.590468       8.266554
true_happy           8.814846        80.556725           0.210881      10.417545
true_neutral         0.716997         0.801350          96.921127       1.560523
true_sad             8.730494         9.363138           1.223113      80.683258
Test accuracy score: 84.616%

MLPClassifier trained
              predicted_angry  predicted_happy  predicted_neutral  predicted_sad
true_angry          96.246307         1.771404           0.000000       1.982286
true_happy           2.825812        94.643608           0.042176       2.488401
true_neutral         0.210881         0.379587          98.945595       0.463939
true_sad             1.180936         2.572754           0.084353      96.161957
Test accuracy score: 96.499%

BaggingClassifier trained
              predicted_angry  predicted_happy  predicted_neutral  predicted_sad
true_angry          93.378326         3.331928           0.253058       3.036693
true_happy           4.808098        91.564735           0.042176       3.584985
true_neutral         0.210881         0.421763          98.861244       0.506116
true_sad             2.867988         2.994517           0.126529      94.010963
Test accuracy score: 94.454%

This process took 5398.4667956 seconds
```


## 2. Testing the sentiment of a single audio file 

> `use_audio_to_predict.py` predicts the sentiment of a single audio file from the specified emotions 

Configure `predict.json` before running `use_audio_to_predict.py` with the [base configuration](#base-configuration)
Then add:

| Configuration | Description | 
| --- | --- | 
| model | classifier used to predict |
| audio | directory to predict the sinfgle audio file |

example config:

```.json
{
    "comment_general": "for general predictions, training, testing",
    "model"     :"MLPClassifier",
    "model_ver" :"16k_3feat",
    "emotions"  :"angry,happy,neutral,sad",
    "features"  :"mfcc,chroma,mel",
    "frequency" :"16k",

    "comment_single": "for single predictions",
    "audio"     :"predict_from_audio/emotion testing audio 16k/angry/a1_high_Dervla_emottsangry_0376.wav",

    "comment_multiple": "for multiple predictions",
    "output"    :""
}
```

Running `use_audio_to_predict.py` with this configuration gives an output in this format:

```.txt
{'angry': 1.2644871494553761e-05, 'happy': 3.876536030931289e-05, 'neutral': 0.9570777500193833, 'sad': 0.04287083974881276}

first prediction  : neutral
second prediction : sad
difference is 91.42069102705706 %

Time it took to predict: 0.06672429999999974 s
```


## 3. Testing the sentiment of multiple audio files
> `use_audio_to_predict_multiple.py` predicts multriple audio files for the specified emotions 

> `get_prediction_time.py` creates a visualisation of the performance of the model that is selected predicting the specified emotions.

Please go into the folder `predict_from_audio/emotion testing audio {frequency to predict at}` and put audio representing that emotion in the corresponding folder. (There are already 40 audio files available in various emotions)
        
    ├── ...
    ├── predict_from_audio                    
    │   ├── emotion testing audio 44k          
    │       ├── angry         
    |       ├── boredom
    |       ├── disgust
    |       ├── fear
    |       ├── happy
    |       ├── neutral
    |       ├── pleasant suprise
    |       ├── sad
    │       └── ...
    │   └── ...  
    └── ...      
    

Configure `predict.json` before running `use_audio_to_predict.py` with the [base configuration](#base-configuration)
Then add:

| Configuration | Description | 
| --- | --- | 
| model | classifier used to predict |
| output | If "excel" then the  output for the emotion probability distribution is printed to an excel file which is saved in `predict_from_audio/emotion testing audio 44k/predictions.xlsx`( so it is easy to copy to another spreadsheet, need to pip install openpyxl for this. **otherwise** The distributions are recorded in a .txt file in `predict_from_audio/emotion testing audio 44k/predictions.txt`|

example config:

```.json
{
    "comment_general": "for general predictions, training, testing",
    "model"     :"MLPClassifier",
    "model_ver" :"16k_3feat",
    "emotions"  :"angry,happy,neutral,sad",
    "features"  :"mfcc,chroma,mel",
    "frequency" :"16k",

    "comment_single": "for single predictions",
    "audio"     :"",

    "comment_multiple": "for multiple predictions",
    "output"    :"excel"
}

```
running `get_prediction_time.py` gives an output in the format:

![16k MLP](https://user-images.githubusercontent.com/80789350/148462672-a5051cdf-5b1a-4302-ac8a-4cdc501ab30c.PNG)

running `use_audio_to_predict_multiple.py` gives saves a file in `predict_from_audio/emotion testing audio 44k/predictions.txt` in the format :

![table](https://user-images.githubusercontent.com/80789350/148704591-24b21b53-2f7a-4cd4-91ed-02133dde2868.PNG)


