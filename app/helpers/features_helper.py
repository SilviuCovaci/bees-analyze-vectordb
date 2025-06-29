import librosa
import librosa.display
import numpy as np

stft_log = None
n_fft = 2048
hop_length = 512
n_mfcc = 20
n_mels = 128

def extract_mel(audio_sample, sr, feat_no = None):
    feat_no  = n_mels if feat_no  is None else feat_no
    return librosa.feature.melspectrogram(y=audio_sample, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels = feat_no)

def extract_mfcc(audio_sample, sr, feat_no=None, pre_emphasis_coeff=None):
    feat_no = n_mfcc if feat_no is None else feat_no

    if pre_emphasis_coeff is not None:
        if pre_emphasis_coeff is True:
            pre_emphasis_coeff = 0.97 
        #pre-emphasis: y[n] = x[n] - a * x[n-1]
        #emphasized = np.append(audio_sample[0], audio_sample[1:] - pre_emphasis_coeff * audio_sample[:-1])
        #audio_sample = emphasized
        audio_sample = librosa.effects.preemphasis(audio_sample, coef=pre_emphasis_coeff)

    return librosa.feature.mfcc(y=audio_sample, sr=sr, n_mfcc=feat_no)
    
def extract_combined(audio_sample, sr, use_rms = True, use_zcr=True, use_spectral_centroids=True, use_spectral_rolloff=True, n_mfcc = 13, pitch_classes = 'all'):
    """
    Extract various audio features from an audio signal.
    
    Parameters:
    y (np.ndarray): Audio time series
    sr (int): Sampling rate
    
    Returns:
    dict: Dictionary of extracted features
    """
    features = {}
    
    # Time-domain features
    if (use_rms):
        features['rms'] = librosa.feature.rms(y=audio_sample).mean()
    if (use_zcr):
        features['zcr'] = librosa.feature.zero_crossing_rate(audio_sample).mean()
    
    # Frequency-domain features
    if (use_spectral_centroids):
        spectral_centroid = librosa.feature.spectral_centroid(y=audio_sample, sr=sr)
        features['spectral_centroid'] = spectral_centroid.mean()
        
    if (use_spectral_rolloff):
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_sample, sr=sr)
        features['spectral_rolloff'] = spectral_rolloff.mean()
    
    # MFCCs
    if (n_mfcc > 0):
        mfccs = librosa.feature.mfcc(y=audio_sample, sr=sr, n_mfcc=n_mfcc)
        for i, mfcc in enumerate(mfccs):
            features[f'mfcc_{i+1}'] = mfcc.mean()
    
    # Chroma features
    if (pitch_classes == 'all'):
        pitch_classes = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    
    if (pitch_classes):
        chroma = librosa.feature.chroma_stft(y=audio_sample, sr=sr)
        for i, pitch_class in enumerate(pitch_classes):
            features[f'chroma_{pitch_class}'] = chroma[i].mean()
    
    return features

def extract_combined1(audio_sample, sr):
    return extract_combined(audio_sample, sr)

def extract_features_from_audio(path_audio_sample, start_time = 0, duration = False, features = []):
    if (duration):
        audio_sample, sr = librosa.load(path_audio_sample, offset=start_time, duration=duration, sr=None)
    else:
        audio_sample, sr = librosa.core.load(path_audio_sample, sr=None)



    extracted_features = {}
    for feature in features:
        parts = feature.split('_', 1)  
        if len(parts) == 1:
            parts.append(None)

        feature_name = parts[0]
        feature_option = parts[1]
        if (not feature_option is None):
            feature_option = int(feature_option)
        if (feature_name == 'stft'):            
            stft = librosa.stft(y=audio_sample, n_fft=n_fft, hop_length=hop_length)
            stft_mag, _ = librosa.magphase(stft)
            feature_data = librosa.amplitude_to_db(stft_mag, ref=np.max)            
        elif (feature_name == 'mel'):
            feature_data = extract_mel(audio_sample=audio_sample, sr=sr, feat_no = feature_option)
        elif (feature_name == 'mfcc'):            
            feature_data = extract_mfcc(audio_sample=audio_sample, sr=sr, feat_no = feature_option)      
        elif (feature_name == 'pe-mfcc'):            
            feature_data = extract_mfcc(audio_sample=audio_sample, sr=sr, feat_no = feature_option, pre_emphasis_coeff=True)         
        elif (feature_name == 'mfcc2'):        
            melspec = extracted_features.get('mel', None)
            melspec = melspec if not melspec is None else  extract_mel(audio_sample=audio_sample, sr=sr)            
            feature_data = librosa.feature.mfcc(S=librosa.power_to_db(melspec), sr=sr, n_mfcc=n_mfcc)
        elif (feature_name == 'delta' or feature_name == 'delta2'):    
            mfcc = extracted_features.get(f'mfcc_{feature_option}', None)
            mfcc = mfcc if not mfcc is None else  extract_mfcc(audio_sample=audio_sample, sr=sr, feat_no = feature_option)
            feature_data = librosa.feature.delta(mfcc, order = 1 if feature_name == 'delta' else 2)  
        elif (feature_name == 'zcr'):  
            feature_data = librosa.feature.zero_crossing_rate(y=audio_sample, frame_length=n_fft, hop_length=hop_length)  
        elif (feature_name == 'rms'):  
            feature_data = librosa.feature.rms(y=audio_sample, frame_length=n_fft, hop_length=hop_length)      
        elif (feature_name == 'speccen'):
            feature_data = librosa.feature.spectral_centroid(y=audio_sample, sr=sr, n_fft=n_fft, hop_length=hop_length)
        elif (feature_name == 'specrol'):
            feature_data = librosa.feature.spectral_rolloff(y=audio_sample, sr=sr, n_fft=n_fft, hop_length=hop_length)
        elif (feature_name == 'chroma'):  
            feature_data = librosa.feature.chroma_stft(y=audio_sample, sr=sr, n_fft=n_fft, hop_length=hop_length)
        extracted_features[feature] = feature_data
    
    return extracted_features