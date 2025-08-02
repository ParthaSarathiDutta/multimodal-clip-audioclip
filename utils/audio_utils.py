import torchaudio
import torchaudio.transforms as T

def load_audio(filepath, sample_rate=16000):
    waveform, sr = torchaudio.load(filepath)
    if sr != sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=sample_rate)
        waveform = resampler(waveform)
    return waveform.squeeze(0)  # Return mono

def get_log_mel_spectrogram(waveform, sample_rate=16000, n_mels=64):
    mel_spectrogram = T.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=1024,
        hop_length=512,
        n_mels=n_mels
    )(waveform)
    return T.AmplitudeToDB()(mel_spectrogram)
