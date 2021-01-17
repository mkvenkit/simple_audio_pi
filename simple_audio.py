"""
    simple_audio.py

    This programs collects audio data from an I2S mic on the Raspberry Pi 
    and runs the TensorFlow Lite interpreter on a per-build model. 


    Author: Mahesh Venkitachalam
    Website: electronut.in

"""

from scipy.io import wavfile
from scipy import signal
import numpy as np
import argparse 
import pyaudio
import wave
import time

from tflite_runtime.interpreter import Interpreter

from display_ssd1306 import SSD1306_Display

VERBOSE_DEBUG = False

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

def get_live_input(disp):
    CHUNK = 4096
    FORMAT = pyaudio.paInt32
    CHANNELS = 2
    RATE = 16000 
    RECORD_SECONDS = 3
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

    # discard first 1 second
    for i in range(0, NFRAMES):
        data = stream.read(CHUNK, exception_on_overflow = False)

    try:
        while True:
            print("Listening...")
            disp.show_txt(0, 0, "Listening...", False)

            frames = []
            for i in range(0, NFRAMES):
                data = stream.read(CHUNK, exception_on_overflow = False)
                frames.append(data)

            # process data
            # 4096 * 3 frames * 2 channels * 4 bytes = 98304 bytes 
            # CHUNK * NFRAMES * 2 * 4 
            buffer = b''.join(frames)
            audio_data = np.frombuffer(buffer, dtype=np.int32)
            nbytes = CHUNK * NFRAMES 
            # reshape for input 
            audio_data = audio_data.reshape((nbytes, 2))
            # run inference on audio data 
            run_inference(disp, audio_data)
    except KeyboardInterrupt:
        print("exiting...")
           
    stream.stop_stream()
    stream.close()
    p.terminate()

def process_audio_data(waveform):
    """Process audio input.

    This function takes in raw audio data from a WAV file and does scaling 
    and padding to 16000 length.

    """

    if VERBOSE_DEBUG:
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # if stereo, pick the left channel
    if len(waveform.shape) == 2:
        print("Stereo detected. Picking one channel.")
        waveform = waveform.T[1]
    else: 
        waveform = waveform 

    if VERBOSE_DEBUG:
        print("After scaling:")
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # normalise audio
    wabs = np.abs(waveform)
    wmax = np.max(wabs)
    waveform = waveform / wmax

    PTP = np.ptp(waveform)
    print("peak-to-peak: %.4f. Adjust as needed." % (PTP,))

    # return None if too silent 
    if PTP < 0.5:
        return []

    if VERBOSE_DEBUG:
        print("After normalisation:")
        print("waveform:", waveform.shape, waveform.dtype, type(waveform))
        print(waveform[:5])

    # scale and center
    waveform = 2.0*(waveform - np.min(waveform))/PTP - 1

    # extract 16000 len (1 second) of data   
    max_index = np.argmax(waveform)  
    start_index = max(0, max_index-8000)
    end_index = min(max_index+8000, waveform.shape[0])
    waveform = waveform[start_index:end_index]

    # Padding for files with less than 16000 samples
    if VERBOSE_DEBUG:
        print("After padding:")

    waveform_padded = np.zeros((16000,))
    waveform_padded[:waveform.shape[0]] = waveform

    if VERBOSE_DEBUG:
        print("waveform_padded:", waveform_padded.shape, waveform_padded.dtype, type(waveform_padded))
        print(waveform_padded[:5])

    return waveform_padded

def get_spectrogram(waveform):
    
    waveform_padded = process_audio_data(waveform)

    if not len(waveform_padded):
        return []

    # compute spectrogram 
    f, t, Zxx = signal.stft(waveform_padded, fs=16000, nperseg=255, 
        noverlap = 124, nfft=256)
    # Output is complex, so take abs value
    spectrogram = np.abs(Zxx)

    if VERBOSE_DEBUG:
        print("spectrogram:", spectrogram.shape, type(spectrogram))
        print(spectrogram[0, 0])
        
    return spectrogram


def run_inference(disp, waveform):

    # get spectrogram data 
    spectrogram = get_spectrogram(waveform)

    if not len(spectrogram):
        #disp.show_txt(0, 0, "Silent. Skip...", True)
        print("Too silent. Skipping...")
        #time.sleep(1)
        return 

    spectrogram1= np.reshape(spectrogram, (-1, spectrogram.shape[0], spectrogram.shape[1], 1))
    
    if VERBOSE_DEBUG:
        print("spectrogram1: %s, %s, %s" % (type(spectrogram1), spectrogram1.dtype, spectrogram1.shape))

    # load TF Lite model
    interpreter = Interpreter('simple_audio_model_numpy.tflite')
    interpreter.allocate_tensors()
    
    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #print(input_details)
    #print(output_details)

    input_shape = input_details[0]['shape']
    input_data = spectrogram1.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    print("running inference...")
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    yvals = output_data[0]
    commands = ['go', 'down', 'up', 'stop', 'yes', 'left', 'right', 'no']

    if VERBOSE_DEBUG:
        print(output_data[0])
    print(">>> " + commands[np.argmax(output_data[0])].upper())
    disp.show_txt(0, 12, commands[np.argmax(output_data[0])].upper(), True)
    #time.sleep(1)

def main():

    # create parser
    descStr = """
    This program does ML inference on audio data.
    """
    parser = argparse.ArgumentParser(description=descStr)
    # add a mutually exclusive group of arguments
    group = parser.add_mutually_exclusive_group()

    # add expected arguments
    group .add_argument('--input', dest='wavfile_name', required=False)
    
    # parse args
    args = parser.parse_args()

    disp = SSD1306_Display()
    
    # test WAV file
    if args.wavfile_name:
        wavfile_name = args.wavfile_name
        # get audio data 
        rate, waveform = wavfile.read(wavfile_name)
        # run inference
        run_inference(disp, waveform)
    else:
        get_live_input(disp)

    print("done.")

# main method
if __name__ == '__main__':
    main()
