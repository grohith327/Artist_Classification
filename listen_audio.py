import pyaudio
import wave
from keras.models import load_model
import librosa
import numpy as np
import warnings
from skimage.feature import peak_local_max

warnings.filterwarnings(action='ignore',category=FutureWarning)

CHUNK = 256
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22050
RECORD_SECONDS = 2
WAVE_OUTPUT_FILENAME = "test.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK, exception_on_overflow = False)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

model_1 = load_model('peaks_total_conv.h5')
# model_2 = load_model('peaks_part1_conv.h5')
# model_3 = load_model('peaks_part2_conv.h5')
print('* models loaded')

data, sf = librosa.load('test.wav')
D = np.abs(librosa.stft(data))
coordinates = peak_local_max(D)
feat_img = np.zeros((1025,216))
feat_img[coordinates[:,0],coordinates[:,1]] = 1

feat_img = feat_img.reshape((1,1025,216,1))

output_1 = model_1.predict(feat_img)
# output_2 = model_2.predict(feat_img)
# output_3 = model_3.predict(feat_img)
print(output_1)
print(np.argmax(output_1))
# print(output_2)
# print(np.argmax(output_2))
# print(output_3)
# print(np.argmax(output_3))

if(np.argmax(output_1) == 0):
	print('Armin')
elif(np.argmax(output_1) == 1):
	print("Bach")
elif(np.argmax(output_1) == 2):
	print("Beatles") 
elif(np.argmax(output_1) == 3):
	print("Iron Maiden")
else:
	print("Kendrick Lamar") 

