from tracemalloc import start
from emotion_recognition import EmotionRecognizer

import pyaudio
import librosa
import json
import wave
import os
import time
from sys import byteorder
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier

from utils import get_grid_tuned_models,string_into_list

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 10

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.
    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > SILENCE:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()

    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, r

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()

def record_without_file():
    sample_width, data = record()
    return(data, sample_width)

def record_to_file_check(original_data, sample_width, path):
    data = pack('<' + ('h'*len(original_data)), *original_data)
    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    

def get_grid_tuned_models_dict(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}



if __name__ == "__main__":
    grid_tuned_models = get_grid_tuned_models(True)
    grid_tuned_models_name, grid_tuned_models_dict = get_grid_tuned_models_dict(grid_tuned_models)

    json_file = "test_train_config.json"
    
    with open(json_file, 'r') as config_file:
        data = json.load(config_file)
        mandatory_settings =    data["MANDATORY FIELD SETTING"]
        classifier_name =       mandatory_settings["classifier name"].format(grid_tuned_models_name)
        model_folder =          mandatory_settings["pre-saved model folder"]
        emotions =              string_into_list(mandatory_settings['emotions'])
        features =              string_into_list(mandatory_settings["features"])
        model_dir = os.path.join(model_folder,classifier_name)
   
    detector = EmotionRecognizer(emotions=emotions, model_dir = model_dir, features=features, verbose=0)

    print("Please talk")
    
    filename = "test.wav"
    data, sample_width  = record_without_file()

    start_predict = time.perf_counter()
    record_to_file_check(data, sample_width,filename)
    end_predict = time.perf_counter()
    
    extra_time_recording_file = end_predict-start_predict

    print(f'\nLength of audio recorded: {librosa.get_duration(filename = filename)} seconds')

# audio buffer directly
    print(f'\nPredicting from recording directily')
    start_predict = time.perf_counter()
    result = detector.predict_proba_audio(data)
    end_predict = time.perf_counter()

    print(f"\n emotion probabilities \n{result}")

    maximum = max(result, key=result.get)
    max_value = result[maximum]
    del result[maximum]

    second = max(result, key=result.get)
    second_value = result[second]

    print(f"\nfirst prediction  : {maximum} \nsecond prediction : {second} \ndifference is {(max_value - second_value)*100} %")

    print(f"\nTime it took to predict: {(end_predict - start_predict)*1000}ms")

# from file
    # print(f'\nPredicting from recording saved to file')
    # start_predict = time.perf_counter()
    # result = detector.predict_proba(filename)
    # end_predict = time.perf_counter()

    # print(f"\n emotion probabilities \n{result}")
    
    # maximum = max(result, key=result.get)
    # max_value = result[maximum]
    # del result[maximum]

    # second = max(result, key=result.get)
    # second_value = result[second]

    # print(f"\nfirst prediction  : {maximum} \nsecond prediction : {second} \ndifference is {(max_value - second_value)*100} %")

    # print(f"\nTime it took to predict: {((end_predict - start_predict) + extra_time_recording_file)*1000} s")


