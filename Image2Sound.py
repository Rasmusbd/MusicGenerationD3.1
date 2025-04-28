import librosa
import cv2
import numpy as np
import soundfile as sf
from pydub import AudioSegment
import os

def Image2Sound(imageNameOrImage, conf):
    
    if type(imageNameOrImage) == str:
        image = cv2.imread(imageNameOrImage, cv2.IMREAD_GRAYSCALE)
        image = image.astype(np.float32) / 255.0
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
        mel_db = image*80.0 - 80.0
        mel_power = librosa.db_to_power(mel_db)
        audio = librosa.feature.inverse.mel_to_audio(mel_power,
                                                sr=conf.sampling_rate,
                                                n_fft=conf.n_fft,
                                                hop_length=conf.hop_length,
                                                fmin=conf.fmin,
                                                fmax=conf.fmax)
        return audio
    
def SaveAudio(audio_array, path, filename, sr=22050):

    os.makedirs(path, exist_ok=True)
    if not filename.endswith('.mp3'):
        filename += '.mp3'

    #Save as .wav wavefile
    temp_wav = os.path.join(path, "temp_output.wav")
    final_mp3 = os.path.join(path, filename)
    sf.write(temp_wav, audio_array, sr)

    #Convert to mp3 and remove the saved wave
    audio_segment = AudioSegment.from_wav(temp_wav)
    audio_segment.export(final_mp3, format="mp3")
    os.remove(temp_wav)



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


audio = Image2Sound("000002.jpg", conf)
SaveAudio(audio,os.getcwd(),"testfile2.mp3")






