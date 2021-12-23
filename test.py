from emotion_recognition import EmotionRecognizer

import pyaudio
import json
import wave
from sys import byteorder
from array import array
from struct import pack
from sklearn.ensemble import GradientBoostingClassifier, BaggingClassifier
import time
import os

from utils import get_best_estimators

THRESHOLD = 500
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

SILENCE = 3

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

def recording_to_file(sample_width, data, path):
    "Records from the microphone and outputs the resulting data to 'path'"
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()


def get_estimators_name(estimators):
    result = [ '"{}"'.format(estimator.__class__.__name__) for estimator, _, _ in estimators ]
    return ','.join(result), {estimator_name.strip('"'): estimator for estimator_name, (estimator, _, _) in zip(result, estimators)}



if __name__ == "__main__":

    estimators = get_best_estimators(True)
    estimators_str, estimator_dict = get_estimators_name(estimators)
    print(estimators_str)

    with open('predict.json') as config_file:
        data = json.load(config_file)
        model =     data["model"].format(estimators_str)
        model_ver = data["model_ver"]
        emotions =  data['emotions'].split(",")
        features =  data["features"].split(",")

    model_name = os.path.join(model_ver,model)
    
    
    detector = EmotionRecognizer(estimator_dict[model] , emotions=emotions, model_name = model_name, features=features, verbose=0)
    
    print("Please talk")
    sample_width, data = record()

    filename = "test.wav"
    recording_to_file(sample_width, data, filename)
    start_predict = time.perf_counter()
    result_file = detector.predict_proba(filename)
    end_predict = time.perf_counter()

    # start_predict = time.perf_counter()
    #result_audio = detector.predict_proba_audio(data)
    # end_predict = time.perf_counter()

    print(f"{result_file}")
    #print(f"from audio {result_audio}")

    maximum = max(result_file, key=result_file.get)
    max_value = result_file[maximum]
    del result_file[maximum]

    second = max(result_file, key=result_file.get)
    second_value = result_file[second]

    print(f"\nfirst prediction  : {maximum} \nsecond prediction : {second} \ndifference is {(max_value - second_value)*100} %")

    print(f"\nTime it took to predict: {end_predict - start_predict} s")