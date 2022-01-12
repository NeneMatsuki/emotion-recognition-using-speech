import os
folder = "predict_from_audio/emotion testing audio 44k/neutral"
files = os.listdir(folder)


for file in files:
    senti,person = file.split("_")
    #os.rename(folder + "/" + files[i], folder + "/" + name + "_happy." + wave)
    os.rename(folder + "/" + file, f"{folder}/{senti}_med_{person}")