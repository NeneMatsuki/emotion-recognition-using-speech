import soundfile
import librosa
import numpy as np
import pickle
import os
from convert_wavs import convert_audio
import json
import sys


AVAILABLE_EMOTIONS = {
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fear",
    "disgust",
    "ps", # pleasant surprised
    "boredom"
}


def get_label(audio_config):
    """Returns label corresponding to which features are to be extracted
        e.g:
    audio_config = {'mfcc': True, 'chroma': True, 'contrast': False, 'tonnetz': False, 'mel': False}
    get_label(audio_config): 'mfcc-chroma'
    """
    features = ["mfcc", "chroma", "mel", "contrast", "tonnetz"]
    label = ""
    for feature in features:
        if audio_config[feature]:
            label += f"{feature}-"
    return label.rstrip("-")


def get_dropout_str(dropout, n_layers=3):
    if isinstance(dropout, list):
        return "_".join([ str(d) for d in dropout])
    elif isinstance(dropout, float):
        return "_".join([ str(dropout) for i in range(n_layers) ])


def get_first_letters(emotions):
    return "".join(sorted([ e[0].upper() for e in emotions ]))


def extract_feature(file_name, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")
    try:
        with soundfile.SoundFile(file_name) as sound_file:
            pass
    except RuntimeError:
        # not properly formated, convert to 16000 sample rate & mono channel using ffmpeg
        # get the basename
        basename = os.path.basename(file_name)
        dirname  = os.path.dirname(file_name)
        name, ext = os.path.splitext(basename)
        new_basename = f"{name}_c.wav"
        new_filename = os.path.join(dirname, new_basename)
        v = convert_audio(file_name, new_filename)
        if v:
            raise NotImplementedError("Converting the audio files failed, make sure `ffmpeg` is installed in your machine and added to PATH.")
    else:
        new_filename = file_name
    with soundfile.SoundFile(new_filename) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma or contrast:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))
    return result

def extract_feature_audio(audio, **kwargs):
    """
    Extract feature from audio file `file_name`
        Features supported:
            - MFCC (mfcc)
            - Chroma (chroma)
            - MEL Spectrogram Frequency (mel)
            - Contrast (contrast)
            - Tonnetz (tonnetz)
        e.g:
        `features = extract_feature(path, mel=True, mfcc=True)`
    """
    mfcc = kwargs.get("mfcc")
    chroma = kwargs.get("chroma")
    mel = kwargs.get("mel")
    contrast = kwargs.get("contrast")
    tonnetz = kwargs.get("tonnetz")

    X = audio
    sample_rate = 16000
    if chroma or contrast:
        stft = np.abs(librosa.stft(X))
    result = np.array([])
    if mfcc:
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
    if chroma:
        chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, chroma))
    if mel:
        mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
        result = np.hstack((result, mel))
    if contrast:
        contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
        result = np.hstack((result, contrast))
    if tonnetz:
        tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
        result = np.hstack((result, tonnetz))
    return result


def get_grid_tuned_models(classification):
    """
    Loads the estimators that are pickled in `grid` folder
    Note that if you want to use different or more estimators,
    you can fine tune the parameters in `grid_search.py` script
    and run it again ( may take hours )
    """
    if classification:
        return pickle.load(open("grid/best_classifiers.pickle", "rb"))
    else:
        return pickle.load(open("grid/best_regressors.pickle", "rb"))


def get_audio_config(features_list):
    """
    Converts a list of features into a dictionary understandable by
    `data_extractor.AudioExtractor` class
    """
    audio_config = {'mfcc': False, 'chroma': False, 'mel': False, 'contrast': False, 'tonnetz': False}
    for feature in features_list:
        if feature not in audio_config:
            raise TypeError(f"Feature passed: {feature} is not recognized.")
        audio_config[feature] = True
    return audio_config
    
def string_into_list(string):
    if(string[0] == "["):
        string = string[1:]
    if(string[-1] == "]"):
        string = string[:-1]
    return(string.split(","))

def load_mandatory_settings(json_file):
    with open(json_file, 'r') as config_file:
        data = json.load(config_file)

        # check if the config file has correct fields
        if(list(data) != ['MANDATORY FIELD SETTING', 'TEST FIELD SETTING', 'TRAIN FIELD SETTING']):
            print((f"Some fields were deleted in the config file {json_file}, it should include 'MANDATORY FIELD SETTING', 'TEST FIELD SETTING', 'TRAIN FIELD SETTING' "))
            sys.exit(f"Some fields were deleted in the config file {json_file}, it should include 'MANDATORY FIELD SETTING', 'TEST FIELD SETTING', 'TRAIN FIELD SETTING' ")

        # check if the mandatory settings has correct fields
        mandatory_settings =    data["MANDATORY FIELD SETTING"]
        if(list(mandatory_settings) != ['classifier name', 'pre-saved model folder', 'emotions', 'features', 'Test or Train', 'FIELD DESCRIPTION']):
            print((f"Some fields were deleted in MANDATORY FIELD SETTING, it should include ['classifier name', 'pre-saved model folder', 'emotions', 'features', 'Test or Train', 'FIELD DESCRIPTION'] "))
            sys.exit(f"Some fields were deleted in MANDATORY FIELD SETTING, it should include ['classifier name', 'pre-saved model folder', 'emotions', 'features', 'Test or Train', 'FIELD DESCRIPTION'] ")

        # check if train or test, if not then exit
        Test_or_train_mode =    mandatory_settings["Test or Train"].lower()
        if((Test_or_train_mode!= "train") and (Test_or_train_mode != "test")) :
            print("Please choose whether to Test or Train.\n This can be done under Mandatory Settings in predict.json")
            sys.exit("Please choose whether to Test or Train.\n This can be done under Mandatory Settings in predict.json")       
    
        # load mandatory settings
        classifier_name =       mandatory_settings["classifier name"]
        model_folder =          mandatory_settings["pre-saved model folder"]
        emotions =              string_into_list(mandatory_settings['emotions'])
        features =              string_into_list(mandatory_settings["features"])
        model_dir = os.path.join(model_folder,classifier_name)
    
    return(Test_or_train_mode, classifier_name, model_folder, emotions, features, model_dir)

def load_testing_settings(json_file):
    with open(json_file, 'r') as config_file:
        data = json.load(config_file)

        test_settings = data["TEST FIELD SETTING"]
        if(list(test_settings) != ['Test mode', 'TEST SINGLE SETTING', 'TEST MULTIPLE SETTING', 'FIELD DESCRIPTION']):
            print((f"Some fields were deleted in TEST FIELD SETTING, it should include ['Test mode', 'TEST SINGLE SETTING', 'TEST MULTIPLE SETTING', 'FIELD DESCRIPTION'] "))
            sys.exit(f"Some fields were deleted in TEST FIELD SETTING, it should include ['Test mode', 'TEST SINGLE SETTING', 'TEST MULTIPLE SETTING', 'FIELD DESCRIPTION'] ")

        test_mode =     test_settings["Test mode"].lower()

        if(test_mode == 'single'):

            # load settings 
            single_settings = test_settings["TEST SINGLE SETTING"]
            if(list(single_settings) != ["Audio directory"]):
                print((f"Field 'Audio directory' needs to be in TEST SINGLE SETTING in {json_file}"))
                sys.exit(f"Field 'Audio directory' needs to be in TEST SINGLE SETTING in {json_file}")

            audio = single_settings["Audio directory"]

            return(test_mode, audio)
        
        elif(test_mode == 'multiple'):
            multiple_settings = test_settings["TEST MULTIPLE SETTING"]
            if(list(multiple_settings) != ["Display predictions", "Plot time taken"]):
                print((f"Fields [Display predictions, Plot time taken] needs to be in TEST SINGLE SETTING in {json_file}"))
                sys.exit(f"Fields [Display predictions, Plot time taken] needs to be in TEST SINGLE SETTING in {json_file}")
                
            display_predictions = multiple_settings["Display predictions"]
            plot_time = multiple_settings["Plot time taken"]

            return(test_mode, display_predictions, plot_time)
        else:
            sys.exit("Please choose whether to predict single or multiple.\n This can be done under Testing Settings, Test mode in predict.json")

def load_training_settings(json_file):
    with open(json_file, 'r') as config_file:
        data = json.load(config_file)
        train_setting = data["TRAIN FIELD SETTING"]

        if(list(train_setting) != ["Train mode", "TRAIN MULTIPLE SETTING", "SECTION DESCRIPTION"]):
            print(f"Fields [Train mode, TRAIN MULTIPLE SETTING, SECTION DESCRIPTION] needs to be in TRAIN FIELD SETTING in {json_file}")
            sys.exit(f"Fields [Train mode, TRAIN MULTIPLE SETTING, SECTION DESCRIPTION] needs to be in TRAIN FIELD SETTING in {json_file}")
            
        train_mode = train_setting["Train mode"].lower()
        if(train_mode == "multiple"):
            multiple_setting = train_setting['TRAIN MULTIPLE SETTING']

            if(list(multiple_setting) != ["Multiple classifiers to train"]):
                print(f"Field [Multiple classifiers to train] needs to be in TRAIN MULTIPLE SETTING in {json_file}")
                sys.exit(f"Field [Multiple classifiers to train] needs to be in TRAIN MULTIPLE SETTING in {json_file}")
            train_classifiers = multiple_setting['Multiple classifiers to train']
            return(train_mode, train_classifiers)
        
        elif(train_mode == "single"):
            return(train_mode)
        
        else:
            sys.exit(f"Please choose whether to train single or multiple in TRAIN FIELD SETTING in {json_file}")
        

