import pyaudio
import numpy as np

import wave

CHUNK = 1000
FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 16000 
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "test.wav"
NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)
# initialize portaudio
p = pyaudio.PyAudio()

# Get input device number
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
for i in range(0, numdevices):
    if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
        print("Input Device id ", i, " - ", p.get_device_info_by_host_api_device_index(0, i).get('name'))

stream = p.open(format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index = 2)

frames = []

# discard first 1 second
data = stream.read(CHUNK)
for i in range(0, NFRAMES):
    pass

print("start recording!")

for i in range(0, NFRAMES):
     data = stream.read(CHUNK)
     #print(data)
     frames.append(data)

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
