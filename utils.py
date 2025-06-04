import librosa
import numpy as np
import tempfile
import os

def extract_features_from_audio(audio_file):
    """업로드된 오디오 파일에서 GTZAN 형식의 특징 추출"""
    try:
        # Streamlit의 UploadedFile 객체를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        # 오디오 로드 (1간)
       y, sr = librosa.load(tmp_file_path, sr=None, duration=60)
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        features = {}
        
        # 1. Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        features['chroma_stft_mean'] = np.mean(chroma_stft)
        features['chroma_stft_var'] = np.var(chroma_stft)
        
        # 2. RMS
        rms = librosa.feature.rms(y=y)
        features['rms_mean'] = np.mean(rms)
        features['rms_var'] = np.var(rms)
        
        # 3. Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = np.mean(spec_cent)
        features['spectral_centroid_var'] = np.var(spec_cent)
        
        # 4. Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = np.mean(spec_bw)
        features['spectral_bandwidth_var'] = np.var(spec_bw)
        
        # 5. Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['rolloff_mean'] = np.mean(rolloff)
        features['rolloff_var'] = np.var(rolloff)
        
        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        features['zero_crossing_rate_mean'] = np.mean(zcr)
        features['zero_crossing_rate_var'] = np.var(zcr)
        
        # 7. Harmony & Perceptr
        y_harm = librosa.effects.harmonic(y=y)
        features['harmony_mean'] = np.mean(y_harm)
        features['harmony_var'] = np.var(y_harm)
        
        y_perc = librosa.effects.percussive(y=y)
        features['perceptr_mean'] = np.mean(y_perc)
        features['perceptr_var'] = np.var(y_perc)
        
        # 8. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features['tempo'] = float(tempo) if not isinstance(tempo, np.ndarray) else float(tempo[0])
        
        # 9. MFCCs (1-20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            features[f'mfcc{i+1}_mean'] = np.mean(mfccs[i])
            features[f'mfcc{i+1}_var'] = np.var(mfccs[i])
        
        return features
        
    except Exception as e:
        print(f"특징 추출 오류: {e}")
        return None
