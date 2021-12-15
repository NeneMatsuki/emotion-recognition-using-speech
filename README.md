# Edited [emotion recognition using speech](https://github.com/x4nth055/emotion-recognition-using-speech) to test quality of models with pre recorded audio.

Please read [The original README](https://github.com/NeneMatsuki/emotion-recognition-using-speech/blob/master/README_original.md) on instructions on using the model and any requirements. 

Summary of model performance found in this [spreadsheet](https://docs.google.com/spreadsheets/d/1eKX86JusWnL_1YBtDadtsKyx1cQiSuedk0V_xlTiHLw/edit?usp=sharing)

## Introduction
- This repository handles building and training Speech Emotion Recognition System.
- The basic idea behind this tool is to build and train/test a suited machine learning ( as well as deep learning ) algorithm that could recognize and detects human emotions from speech.
- This is useful for many industry fields such as making product recommendations, affective computing, etc.
- Check this [tutorial](https://www.thepythoncode.com/article/building-a-speech-emotion-recognizer-using-sklearn) for more information.
## Requirements
- **Python 3.6+**
### Python Packages
- **tensorflow**
- **librosa==0.6.3**
- **numpy**
- **pandas**
- **soundfile==0.9.0**
- **wave**
- **sklearn**
- **tqdm==4.28.1**
- **matplotlib==2.2.3**
- **pyaudio==0.2.11**
- **openpyxl**
- **[ffmpeg](https://ffmpeg.org/) (optional)**: used if you want to add more sample audio by converting to 16000Hz sample rate and mono channel which is provided in ``convert_wavs.py``

Install these libraries by the following command:
```
pip3 install -r requirements.txt
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

## Testing the sentiment of a single audio file 
Please configure by going into the .json file `.vscode/launch.json`

In the "Python: model prediction" configuration, edit args in such a way that it is formatted as [emotion, model, audio file directory]

```.json
        {
            "name": "Python: model prediction",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "neutral,calm,happy,sad,angry,fear,disgust,ps,boredom",
                "BaggingClassifier",
                "predict_from_audio/emotion testing audio 44k/d1_DC_d08.wav"
            ]
        }

```
Then please run `use_audio_to_predict.py` using this configuration

## Testing the sentiment of multiple audio files

Please go into the folder `predict_from_audio/emotion testing audio 44k` and put audio representing that emotion in the corresponding folder
        
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
    

In the "Python: model prediction" configuration, edit args in such a way that it is formatted as [emotion, model, print to excel]

```.json
        {
            "name": "Python: model prediction",
            "type": "python",
            "request": "launch",
            "program": "${file}",
            "console": "integratedTerminal",
            "args": [
                "neutral,calm,happy,sad,angry,fear,disgust,ps,boredom",
                "BaggingClassifier",
                "yes"
            ]
        }

```

**If printing to excel is yes**, The output for the emotion probability distribution is printed to an excel file which is saved in `predict_from_audio/emotion testing audio 44k/predictions.xlsx`( so it is easy to copy to another spreadsheet, need to pip install openpyxl for this. **otherwise** The distributions are recorded in a .txt file in `predict_from_audio/emotion testing audio 44k/predictions.txt`

Please run `use_audio_to_predict.py` using this configuration

Results of performance of different models are stored here[spreadsheet](https://docs.google.com/spreadsheets/d/1eKX86JusWnL_1YBtDadtsKyx1cQiSuedk0V_xlTiHLw/edit?usp=sharing). |

