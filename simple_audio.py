"""

    simple_audio.py

"""

from scipy.io import wavfile
from scipy import signal
import numpy as np
import argparse 

from tflite_runtime.interpreter import Interpreter

def get_spectrogram(wavfile_name):
    
    rate, waveform = wavfile.read(wavfile_name)
    print("waveform:", waveform.shape, waveform.dtype, type(waveform))
    print(waveform[:5])

    # if stereo, pick the left channel
    if len(waveform.shape) == 2:
        print("Stereo detected. Picking one channel.")
        waveform = waveform.T[1]
    else: 
        waveform = waveform 
    spectrogram = None
        
    print("After scaling:")
    print("waveform:", waveform.shape, waveform.dtype, type(waveform))
    print(waveform[:5])

    # normalise audio
    wabs = np.abs(waveform)
    print("wabs :", wabs.shape)
    print(wabs[:5])
    wmax = np.max(wabs)
    print("wmax = ", wmax)
    waveform = waveform / wmax
    print("After normalisation:")
    print("waveform:", waveform.shape, waveform.dtype, type(waveform))
    print(waveform[:5])

    # Padding for files with less than 16000 samples
    print("After padding:")
    waveform_padded = np.zeros((16000,))
    waveform_padded[:waveform.shape[0]] = waveform
    print("waveform_padded:", waveform_padded.shape, waveform_padded.dtype, type(waveform_padded))
    print(waveform_padded[:5])

    #exit(0)

    f, t, Zxx = signal.stft(waveform_padded, fs=16000, nperseg=255, noverlap = 124, nfft=256)
    spectrogram = np.abs(Zxx)

    

    print("spectrogram:", spectrogram.shape, type(spectrogram))
    print(spectrogram[0, 0])
        
    return waveform, spectrogram

def get_inference(wavfile_name):

    # get spectrogram data 
    waveform, spectrogram = get_spectrogram(wavfile_name)

    spectrogram1= np.reshape(spectrogram, (-1, spectrogram.shape[0], spectrogram.shape[1], 1))
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

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    yvals = output_data[0]
    commands = ['go', 'down', 'up', 'stop', 'yes', 'left', 'right', 'no']
    print(output_data[0])
    print(commands[np.argmax(output_data[0])])
    
    
"""
    prediction = model(spectrogram1)
    print(prediction)
    sm = tf.nn.softmax(prediction[0])
    am = tf.math.argmax(sm)
    print(sm)
    print(commands[am])
"""

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


    # test WAV file
    if args.wavfile_name:
        wavfile_name = args.wavfile_name
    else:
        wavfile_name = 'c1d39ce8_nohash_9.wav'

    # run inference
    get_inference(wavfile_name)

# main method
if __name__ == '__main__':
    main()
