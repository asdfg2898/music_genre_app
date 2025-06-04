import streamlit as st
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import extract_features_from_audio
import os

# 페이지 설정
st.set_page_config(
    page_title="🎵 GTZAN 음악 장르 분류기",
    page_icon="🎵",
    layout="wide"
)

@st.cache_resource
def load_models():
    """저장된 모델과 전처리기들을 로드"""
    try:
        # XGBClassifier 모델 로드 (코랩과 동일하게)
        model = joblib.load('model_xgb.pkl')  # XGBClassifier 객체로 저장된 파일
        
        # 전처리기들 로드
        scaler = joblib.load('scaler_gtzan.pkl')
        label_encoder = joblib.load('label_encoder_gtzan.pkl')
        feature_cols = joblib.load('features_columns.pkl')
        
        return model, scaler, label_encoder, feature_cols
    except Exception as e:
        st.error(f"모델 로드 실패: {e}")
        return None, None, None, None

def predict_genre(audio_file, model, scaler, label_encoder, feature_cols):
    """장르 예측 함수 (코랩 버전과 동일하게 수정)"""
    # 특징 추출
    features = extract_features_from_audio(audio_file)
    if features is None:
        return None, None
    
    # 특징 벡터 생성 (코랩과 동일한 순서)
    feature_vector = []
    try:
        for col in feature_cols:
            if col in features:
                value = features[col]
                if not isinstance(value, (int, float, np.number)):
                    st.error(f"특징 '{col}'의 값이 스칼라가 아님: {type(value)}")
                    return None, None
                feature_vector.append(value)
            else:
                st.error(f"필요한 특징 '{col}'이 추출되지 않았습니다.")
                return None, None
        
        # numpy 배열로 변환 및 reshape
        X = np.array(feature_vector).reshape(1, -1)
        
        # 스케일링
        X_scaled = scaler.transform(X)
        
        # 예측 수행 (XGBClassifier 사용)
        prediction_encoded = model.predict(X_scaled)
        predicted_label_index = int(prediction_encoded[0])
        
        # 확률 예측
        prediction_proba = model.predict_proba(X_scaled)
        
        # 장르명 변환
        genre_name = label_encoder.inverse_transform([predicted_label_index])[0]
        
        return genre_name, prediction_proba[0]
        
    except Exception as e:
        st.error(f"예측 중 오류 발생: {e}")
        return None, None

def main():
    st.title("🎵 GTZAN 음악 장르 분류기")
    st.markdown("### XGBoost 모델로 음악 장르를 예측해보세요! (정확도: 89.24%)")
    
    # 모델 로드
    model, scaler, label_encoder, feature_cols = load_models()
    
    if model is None:
        st.error("모델 파일들을 찾을 수 없습니다. 다음 파일들이 필요합니다:")
        st.code("""
        model_xgb.pkl
        scaler_gtzan.pkl
        label_encoder_gtzan.pkl
        features_columns.pkl
        """)
        return
    
    # 사이드바
    with st.sidebar:
        st.header("📊 프로젝트 정보")
        st.info("""
        **GTZAN 데이터셋 기반**
        - 10개 장르 분류
        - 57개 오디오 특징 사용
        - XGBoost 모델
