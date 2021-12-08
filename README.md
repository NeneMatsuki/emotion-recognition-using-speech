# Edited [emotion recognition using speech](https://github.com/x4nth055/emotion-recognition-using-speech) to test quality of models with pre recorded audio.

Please read [The original README](https://github.com/NeneMatsuki/emotion-recognition-using-speech/blob/master/README_original.md) on instructions on using the model and any requirements. 

Summary of model performance found in this [spreadsheet](https://docs.google.com/spreadsheets/d/1eKX86JusWnL_1YBtDadtsKyx1cQiSuedk0V_xlTiHLw/edit?usp=sharing)

## Data for testing audio

The audio used for testing is provided in the folder `predict_from_audio/emotion testing audio 44k`

### The names of the audio are formatted in the way:

```
emotion key _ intensity (if applicable) _ voice actor _ filename used in original dataset 

eg

a1_high_Dervla_emottsangry_0376.wav
```

Where:

| Emotion key | Meaning | Dataset these emotions were taken from |
| ----------- | ------- | --- |
| a | angry | soul machines deep learning dataset |
| h | happy | soul machines deep learning dataset |
| s | sad | soul machines deep learning dataset |
| n | neutal | soul machines deep learning dataset |
| d | digust | SAVEE dataset |
| su | suprise| SAVEE dataset |
| f | fear |  SAVEE dataset |
| b-n | boring/neutral | soul machines deep learning dataset |



## Configuring model, emotions, and audio file test the audio

Please configure by going into the .json file `.vscode/launch.json`

In the "Python: model prediction" configuration, edit args in such a way that it is formatted as [emotion, model, audio file directory]

For example, below configuration uses the Bagging Classifier model to predict emotions neutral,calm,happy,sad,angry,fear,disgust,ps,boredom from the file predict_from_audio/emotion testing audio 44k/d1_DC_d08.wav

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

## Functions to test models

Using the edited model prediction.json file above, running:
| Function name | Description |
| --- | --- |
| `use_audio_to_predict.py` | Outputs the predicted emotion for the audio file along with probability distribution of audio using the given model |
| `use_audio_to_predict_deep.py` | Outputs the predicted emotion for the audio file along with probability distribution of audio using an RNN model |
| `use_audio_to_predict_multiple.py` | Reads all the audio files with the relevant emotion being tested`predict_from_audio/emotion testing audio 44k` and writes the emotion probability distribution to the file `predict_from_audio/emotion testing audio 44k/predictions.txt`. I used this to make this [spreadsheet](https://docs.google.com/spreadsheets/d/1eKX86JusWnL_1YBtDadtsKyx1cQiSuedk0V_xlTiHLw/edit?usp=sharing) to find the best model. |

