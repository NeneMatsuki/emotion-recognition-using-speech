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

    X = pcm2float(audio, dtype='float32')
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


def get_pre_tuned_models(classification):
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

def get_pre_tuned_models_dict(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}


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
    """
    Make sure the user input which could or could not be a string in a list form in a list
    
    """
    if(string[0] == "["):
        string = string[1:]
    if(string[-1] == "]"):
        string = string[:-1]
    return(string.split(","))

# load input settings

def load_input_settings(json_file):
    """
    Loads mandatory settings from the configuration json file

    """
    with open(json_file, 'r') as config_file:
        data = json.load(config_file)

        # check if the config file has correct fields
        if(list(data) != ['MANDATORY FIELD SETTING', 'TEST FIELD SETTING', 'TRAIN FIELD SETTING']):
            print((f"Some fields were deleted in the config file {json_file}, it should include 'MANDATORY FIELD SETTING', 'TEST FIELD SETTING', 'TRAIN FIELD SETTING' "))
            sys.exit(f"Some fields were deleted in the config file {json_file}, it should include 'MANDATORY FIELD SETTING', 'TEST FIELD SETTING', 'TRAIN FIELD SETTING' ")

        # check if the mandatory settings has correct fields
        mandatory_settings =    data["MANDATORY FIELD SETTING"]
        if(list(mandatory_settings) != ['Model name', 'Pre-saved model folder', 'Emotions', 'Features', 'Test or Train', 'Mode','FIELD DESCRIPTION']):
            print((f"Some fields were deleted in MANDATORY FIELD SETTING, it should include ['Model name', 'Pre-saved model folder', 'Emotions', 'Features', 'Test or Train', 'Mode','FIELD DESCRIPTION'] "))
            sys.exit(f"Some fields were deleted in MANDATORY FIELD SETTING, it should include ['Model name', 'Pre-saved model folder', 'Emotions', 'Features', 'Test or Train', 'Mode','FIELD DESCRIPTION'] ")

        # load mandatory settings
        model_name =            mandatory_settings["Model name"]
        model_folder =          mandatory_settings["Pre-saved model folder"]
        emotions =              string_into_list(mandatory_settings['Emotions'])
        features =              string_into_list(mandatory_settings["Features"])
        model_dir =             os.path.join(model_folder,model_name)

        # Get test/train settings
        test_or_train =    mandatory_settings["Test or Train"].lower()
        mode =             mandatory_settings["Mode"].lower()

        # if test, load test settings
        if(test_or_train == "test"):
            # include if here to check and sys exit
            test_settings = data["TEST FIELD SETTING"]

            # if the config file does not have the correct fields, then exit 
            if(list(test_settings) != ['TEST SINGLE SETTING', 'TEST MULTIPLE SETTING', 'FIELD DESCRIPTION']):
                print((f"Some fields were deleted in TEST FIELD SETTING, it should include ['Test mode', 'TEST SINGLE SETTING', 'TEST MULTIPLE SETTING', 'FIELD DESCRIPTION'] "))
                sys.exit(f"Some fields were deleted in TEST FIELD SETTING, it should include ['Test mode', 'TEST SINGLE SETTING', 'TEST MULTIPLE SETTING', 'FIELD DESCRIPTION'] ")

            # If testing a single audio
            if(mode == 'single'):
                
                # load and check settings 
                single_settings = test_settings["TEST SINGLE SETTING"]
                if(list(single_settings) != ["Audio directory"]):
                    print((f"Field 'Audio directory' needs to be in TEST SINGLE SETTING in {json_file}"))
                    sys.exit(f"Field 'Audio directory' needs to be in TEST SINGLE SETTING in {json_file}")

                # return the audio directory of the single audio 
                return(test_or_train, mode, model_name, model_folder, emotions, features, model_dir, single_settings)
            
            # If testing multiple audio files
            elif(mode == 'multiple'):
                # load and check settings
                multiple_settings = test_settings["TEST MULTIPLE SETTING"]
                if(list(multiple_settings) != ["Display predictions", "Plot stats bool"]):
                    print((f"Fields [Display predictions, Plot time taken] needs to be in TEST SINGLE SETTING in {json_file}"))
                    sys.exit(f"Fields [Display predictions, Plot time taken] needs to be in TEST SINGLE SETTING in {json_file}")

                return(test_or_train, mode, model_name, model_folder, emotions, features, model_dir, multiple_settings)

            # If doing live testing
            elif(mode == 'live'):
                return(test_or_train, mode, model_name, model_folder, emotions, features, model_dir, None)

            # is not specified
            else:
                sys.exit("Please choose whether to predict single or multiple.\n This can be done under Testing Settings, Test mode in predict.json")
        
        # if train load training settings
        elif(test_or_train == "train"):
            
            train_setting = data["TRAIN FIELD SETTING"]

            # if the config file does not have the correct fields, then exit 
            if(list(train_setting) != ["TRAIN MULTIPLE SETTING", "FIELD DESCRIPTION"]):
                print(f"Fields [TRAIN MULTIPLE SETTING, FIELD DESCRIPTION] needs to be in TRAIN FIELD SETTING in {json_file}")
                sys.exit(f"Fields [TRAIN MULTIPLE SETTING, FIELD DESCRIPTION] needs to be in TRAIN FIELD SETTING in {json_file}")
            
            if(mode == "multiple"):
                multiple_setting = train_setting['TRAIN MULTIPLE SETTING']

                # check if fields are correct
                if(list(multiple_setting) != ["Multiple classifiers to train"]):
                    print(f"Field [Multiple classifiers to train] needs to be in TRAIN MULTIPLE SETTING in {json_file}")
                    sys.exit(f"Field [Multiple classifiers to train] needs to be in TRAIN MULTIPLE SETTING in {json_file}")
            
                # get classidfiers to train
                return(test_or_train, mode, model_name, model_folder, emotions, features, model_dir, multiple_setting)
        
            # if train mode is single, don't return anythingS
            elif(mode == "single"):
                return(test_or_train, mode, model_name, model_folder, emotions, features, model_dir, None)
        
            else:
                sys.exit(f"Please choose whether to train single or multiple in TRAIN FIELD SETTING in {json_file}")
        
        # if not either, then exit
        else:
            print("Please choose whether to Test or Train.\n This can be done under Mandatory Settings in predict.json")
            sys.exit("Please choose whether to Test or Train.\n This can be done under Mandatory Settings in predict.json")


def pcm2float(sig, dtype='float32'):
    """Convert PCM signal to floating point with a range from -1 to 1.
    Use dtype='float32' for single precision.
    Parameters
    ----------
    sig : array_like
        Input array, must have integral type.
    dtype : data type, optional
        Desired (floating point) data type.
    Returns
    -------
    numpy.ndarray
        Normalized floating point data.
    See Also
    --------
    float2pcm, dtype
    """
    sig = np.asarray(sig)
    dtype = np.dtype(dtype)

    i = np.iinfo(sig.dtype)
    abs_max = 2 ** (i.bits - 1)
    offset = i.min + abs_max
    return (sig.astype(dtype) - offset) / abs_max
        

