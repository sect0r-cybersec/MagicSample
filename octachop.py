## I don't know what these do, but librosa needs these or it breaks! Dependency nightmare!
import aifc, sunau

import librosa.core, librosa.beat

path = ("test_data/omen.wav")

audio_data, sr = librosa.load(path, sr=22050)
bpm, beats = librosa.beat.beat_track(y=audio_data, sr=sr)
print(bpm)
