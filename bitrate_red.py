import soundfile as sf 
import librosa 
import numpy as np
import os

for dirc in os.listdir('./data/'):
	print("Reading in "+dirc)
	for files in os.listdir('./data/'+str(dirc)):
		print(files)
		y,s = librosa.load('./data/'+str(dirc)+'/'+str(files),sr=8000)
		if(os.path.isdir('./new_data/'+str(dirc)) == False):
			os.mkdir('./new_data/'+str(dirc))
		sf.write('./new_data/'+str(dirc)+'/'+str(files), y, 8000, subtype='PCM_24')