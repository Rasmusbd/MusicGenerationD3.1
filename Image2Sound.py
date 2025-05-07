import librosa
import cv2
import numpy as np
import soundfile as sf
import os
import subprocess

class conf:
    sampling_rate = 44100
    duration = 6
    samples = sampling_rate * duration
    n_mels = 512
    hop_length = 512
    n_fft = 4096
    fmin = 20
    fmax = sampling_rate // 2

def denormalize_spectrogram(norm_spectrogram, global_min=-100.00001, global_max=43.668285):
    return norm_spectrogram * (global_max - global_min) + global_min

def Image2Sound(imageNameOrImage, conf = conf):
    
    if type(imageNameOrImage) == str:
        import matplotlib.pyplot as plt
        image = cv2.imread(imageNameOrImage, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        plt.imshow(image, aspect='auto', origin='lower', cmap="gray")
        plt.colorbar()
        plt.title("Image after denormalization")
        plt.show()
        mel_db = denormalize_spectrogram(image)
        plt.imshow(mel_db, aspect='auto', origin='lower', cmap="gray")
        plt.colorbar()
        plt.title("melDB")
        plt.show()
        mel_power = librosa.db_to_power(mel_db)
        audio = librosa.feature.inverse.mel_to_audio(mel_power,
                                                sr=conf.sampling_rate,
                                                n_fft=conf.n_fft,
                                                hop_length=conf.hop_length,
                                                fmin=conf.fmin,
                                                fmax=conf.fmax)
                                                
        return audio
    else:
        print("Please save the file as and image and use that to do image2sound") 
    
def SaveAudio(audio_array, path, filename, sr=44100):
    os.makedirs(path, exist_ok=True)
    if not filename.endswith('.mp3'):
        filename += '.mp3'

    temp_wav = os.path.join(path, "temp_output.wav")
    final_mp3 = os.path.join(path, filename)

    sf.write(temp_wav, audio_array, sr, subtype='PCM_16')
    subprocess.run(['ffmpeg', '-i', temp_wav, '-q:a', '0', final_mp3], check=True)
    os.remove(temp_wav)




#Test like this, it should reproduce the example song in test_music_folder
"""
audio = Image2Sound("000002_1.jpg", conf)
SaveAudio(audio,os.getcwd(),"testfile.mp3")
"""




