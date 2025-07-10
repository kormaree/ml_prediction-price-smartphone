import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Предсказание цен на смартфоны",
    page_icon="📱",
    layout="wide"
)

@st.cache_resource
def load_model():
    try:
        model = joblib.load('xgb.joblib')
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None

def create_features(data):
    try:
        data['total_pixels'] = data['resolution_width'] * data['resolution_height']
        
        data['high_end'] = (data['has_5g'] & (data['refresh_rate'] > 90)).astype(int)
        
        return data
    except Exception as e:
        st.error(f"Ошибка при создании признаков: {e}")
        return None

def predict_price(model, input_data):
    try:

        df = pd.DataFrame([input_data], columns=[
            'has_nfc', 'has_5g', 'processor_speed', 
            'internal_memory', 'refresh_rate',
            'resolution_width', 'resolution_height'
        ])
        
        df = create_features(df)
        if df is None:
            return None
            
        expected_features = ['has_nfc', 'processor_speed', 'internal_memory', 'total_pixels', 'high_end']
        df = df[expected_features]
        
        imputer = SimpleImputer(strategy='median')
        df_imputed = imputer.fit_transform(df)
        
        log_prediction = model.predict(df_imputed)[0]
        prediction = np.exp(log_prediction)
        
        return round(prediction, 2)
        
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
        return None

def main():
    st.title("📱 Предсказание цен на смартфоны")
    st.markdown("Введите характеристики смартфона для получения предсказания цены")

    model = load_model()
    if model is None:
        st.stop()

    st.sidebar.header("Характеристики смартфона")

    st.sidebar.subheader("📱 Основные параметры")
    has_nfc = st.sidebar.checkbox("Наличие NFC")
    has_5g = st.sidebar.checkbox("Поддержка 5G")

    st.sidebar.subheader("⚡ Производительность")
    processor_speed = st.sidebar.slider(
        "Частота процессора (ГГц)", min_value=1.0, max_value=3.5, value=2.0, step=0.1)

    st.sidebar.subheader("💾 Память и дисплей")
    internal_memory = st.sidebar.slider(
        "Встроенная память (ГБ)", min_value=32, max_value=1024, value=128)
    refresh_rate = st.sidebar.slider(
        "Частота обновления (Гц)", min_value=60, max_value=144, value=60)
    
    st.sidebar.subheader("📷 Камеры")
    resolution_width = st.sidebar.slider(
        "Ширина экрана (пиксели)", min_value=720, max_value=3840, value=1080)
    resolution_height = st.sidebar.slider(
        "Высота экрана (пиксели)", min_value=1280, max_value=2160, value=1920)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Введенные параметры")
        
        params_data = {
            "Параметр": ["NFC", "5G", "Процессор (ГГц)", "Память (ГБ)", "Частота обновления", "Разрешение экрана"],
            "Значение": [ 
                "Да" if has_nfc else "Нет",
                "Да" if has_5g else "Нет",
                f"{processor_speed:.1f}",
                str(internal_memory),
                str(refresh_rate),
                f"{resolution_width}x{resolution_height}",
            ]
        }

        params_df = pd.DataFrame(params_data)
        st.dataframe(params_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Предсказание")

        if st.button("Предсказать цену", type="primary", use_container_width=True):
            input_data = {
                'has_nfc': int(has_nfc),
                'has_5g': int(has_5g),
                'processor_speed': processor_speed,
                'internal_memory': internal_memory,
                'refresh_rate': refresh_rate,
                'resolution_width': resolution_width,
                'resolution_height': resolution_height,
            }

            with st.spinner("Вычисляю предсказание..."):
                predicted_price = predict_price(model, input_data)

            if predicted_price is not None:
                st.metric(
                    label="Предсказанная цена",
                    value=f"${predicted_price:,.0f}",
                    delta=None
                )

            else:
                st.error("Не удалось выполнить предсказание")

if __name__ == "__main__":
    main()