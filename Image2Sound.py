import librosa
import cv2
import numpy as np
import soundfile as sf
import os
import subprocess

class conf:
    # Preprocessing settings
    sampling_rate = 44100
    duration = 30
    hop_length = 694
    fmin = 20
    fmax = sampling_rate // 2
    n_mels = 128
    n_fft = n_mels * 20
    samples = sampling_rate * duration

def Image2Sound(imageNameOrImage, conf):
    
    if type(imageNameOrImage) == str:
        image = cv2.imread(imageNameOrImage, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
        expected_frames = int(np.ceil((conf.samples - conf.n_fft) / conf.hop_length)) + 1
        image = cv2.resize(image, (expected_frames, conf.n_mels))
        mel_db = image*80.0 - 80.0
        mel_power = librosa.db_to_power(mel_db)
        audio = librosa.feature.inverse.mel_to_audio(mel_power,
                                                sr=conf.sampling_rate,
                                                n_fft=conf.n_fft,
                                                hop_length=conf.hop_length,
                                                fmin=conf.fmin,
                                                fmax=conf.fmax)
                                                
        return audio
    else:
        imageNameOrImage = image.astype(np.float32) / 255.0
        mel_db = image*80.0 - 80.0
        mel_power = librosa.db_to_power(mel_db)
        audio = librosa.feature.inverse.mel_to_audio(mel_power,
                                                sr=conf.sampling_rate,
                                                n_fft=conf.n_fft,
                                                hop_length=conf.hop_length,
                                                fmin=conf.fmin,
                                                fmax=conf.fmax)
        return audio
    
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
audio = Image2Sound("000002.jpg", conf)
SaveAudio(audio,os.getcwd(),"testfile.mp3")
"""




