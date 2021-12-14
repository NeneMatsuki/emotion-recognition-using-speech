import os
folder = "data/test-custom"
files = os.listdir(folder)


for i in range(len(files)):
    name,wave = files[i].split(".")
    #os.rename(folder + "/" + files[i], folder + "/" + name + "_happy." + wave)
    os.rename(folder + "/" + files[i], folder + "/" + "Erica_" + files[i])