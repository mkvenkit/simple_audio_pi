import pyaudio
import numpy as np
from scipy.io import wavfile

import wave

# get pyaudio input device
def getInputDevice(p):
    index = None
    nDevices = p.get_device_count()
    print('Found %d devices:' % nDevices)
    for i in range(nDevices):
        deviceInfo = p.get_device_info_by_index(i)
        #print(deviceInfo)
        devName = deviceInfo['name']
        print(devName)
        # look for the "input" keyword
        # choose the first such device as input
        # change this loop to modify this behavior
        # maybe you want "mic"?
        if not index:
            if 'input' in devName.lower():
                index = i
    # print out chosen device
    if index is not None:
        devName = p.get_device_info_by_index(index)["name"]
        #print("Input device chosen: %s" % devName)
    return index

def main():

    CHUNK = 4096
    FORMAT = pyaudio.paInt32
    CHANNELS = 2
    RATE = 16000 
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "test.wav"
    NFRAMES = int((RATE * RECORD_SECONDS) / CHUNK)

    # initialize pyaudio
    p = pyaudio.PyAudio()
    getInputDevice(p)

    print('opening stream...')
    stream = p.open(format = FORMAT,
                    channels = CHANNELS,
                    rate = RATE,
                    input = True,
                    frames_per_buffer = CHUNK,
                    input_device_index = 1)

    frames = []

    # discard first 1 second
    for i in range(0, NFRAMES):
        data = stream.read(CHUNK)

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

# main method
if __name__ == '__main__':
    main()