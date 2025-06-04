import librosa
import numpy as np
import tempfile
import os

def extract_features_from_audio(audio_file):
    """업로드된 오디오 파일에서 GTZAN 형식의 특징 추출 (코랩 버전과 동일)"""
    try:
        # Streamlit의 UploadedFile 객체를 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.read())
            tmp_file_path = tmp_file.name
        
        print(f"오디오 파일 로딩 중: {tmp_file_path}")
        # 오디오 로드 (60초로 변경 - 코랩 버전과 동일)
        y, sr = librosa.load(tmp_file_path, sr=None, duration=60)
        print(f"오디오 로딩 완료. sr={sr}, 길이={len(y)}")
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        features = {}
        
        # 각 특징 계산 후 타입 확인 추가 (코랩 버전과 동일)
        def add_feature(name, value):
            if not isinstance(value, (int, float, np.number)):
                print(f"경고: 특징 '{name}'의 타입이 스칼라가 아님: {type(value)}, 값: {value}")
            features[name] = value
        
        # 1. Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
        add_feature('chroma_stft_mean', np.mean(chroma_stft))
        add_feature('chroma_stft_var', np.var(chroma_stft))
        
        # 2. RMS
        rms = librosa.feature.rms(y=y)
        add_feature('rms_mean', np.mean(rms))
        add_feature('rms_var', np.var(rms))
        
        # 3. Spectral Centroid
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        add_feature('spectral_centroid_mean', np.mean(spec_cent))
        add_feature('spectral_centroid_var', np.var(spec_cent))
        
        # 4. Spectral Bandwidth
        spec_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        add_feature('spectral_bandwidth_mean', np.mean(spec_bw))
        add_feature('spectral_bandwidth_var', np.var(spec_bw))
        
        # 5. Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        add_feature('rolloff_mean', np.mean(rolloff))
        add_feature('rolloff_var', np.var(rolloff))
        
        # 6. Zero Crossing Rate
        zcr = librosa.feature.zero_crossing_rate(y=y)
        add_feature('zero_crossing_rate_mean', np.mean(zcr))
        add_feature('zero_crossing_rate_var', np.var(zcr))
        
        # 7. Harmony (코랩 버전과 동일하게 분리)
        y_harm = librosa.effects.harmonic(y=y)
        add_feature('harmony_mean', np.mean(y_harm))
        add_feature('harmony_var', np.var(y_harm))
        
        # 8. Perceptr (코랩 버전과 동일하게 분리)
        y_perc = librosa.effects.percussive(y=y)
        add_feature('perceptr_mean', np.mean(y_perc))
        add_feature('perceptr_var', np.var(y_perc))
        
        # 9. Tempo (코랩 버전과 동일한 처리)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray) and tempo.size > 0:
            tempo_value = tempo[0]
            print(f"경고: Tempo가 배열로 반환됨. 첫 번째 값 사용: {tempo_value}")
        elif isinstance(tempo, np.ndarray) and tempo.size == 0:
            tempo_value = 120.0
            print(f"경고: Tempo 감지 실패. 기본값 사용: {tempo_value}")
        else:
            tempo_value = tempo
        add_feature('tempo', tempo_value)
        
        # 10. MFCCs (1-20)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        for i in range(20):
            add_feature(f'mfcc{i+1}_mean', np.mean(mfccs[i]))
            add_feature(f'mfcc{i+1}_var', np.var(mfccs[i]))
        
        print("특징 추출 완료.")
        print(f"추출된 특징 개수: {len(features)}")
        return features
        
    except Exception as e:
        print(f"특징 추출 오류: {e}")
        return None
