import os
import numpy as np
import pickle
import streamlit as st
from streamlit_option_menu import option_menu

current_dir = os.path.dirname(os.path.abspath(__file__))
model_file_path = os.path.join(current_dir, 'trained_model_kebayoranBaru.sav')
scale_file_path = os.path.join(current_dir, 'scale_MMS_kebayoranBaru.pkl')

rfr_model = pickle.load(open(model_file_path, 'rb'))
scale_MMS = pickle.load(open(scale_file_path, 'rb'))

def predict_price(input_data):
    
    arr_input    = np.asarray(input_data)
    rsh_input    = arr_input.reshape(1, -1)

    _input       = scale_MMS.transform(rsh_input)
    _input       = _input[:,1:]
    
    pred         = rfr_model.predict(_input)

    pred = pred.reshape(-1, 1)
    pred         = np.concatenate((pred, _input), axis=1)
    final_pred   = scale_MMS.inverse_transform(pred)
    
    return final_pred[0][0]


def rupiah(value):
    str_value = str(value)
    separate_decimal = str_value.split(".")
    after_decimal = separate_decimal[0]
    before_decimal = separate_decimal[1]

    reverse = after_decimal[::-1]
    temp_reverse_value = ""

    for index, val in enumerate(reverse):
        if (index + 1) % 3 == 0 and index + 1 != len(reverse):
            temp_reverse_value = temp_reverse_value + val + "."
        else:
            temp_reverse_value = temp_reverse_value + val

    temp_result = temp_reverse_value[::-1]

    return "Rp " + temp_result + "," + before_decimal


def main():
    
    st.title('Prediksi Harga Properti Kebayoran Baru :chart_with_upwards_trend:')
    st.caption('Selamat datang di Aplikasi Web prediksi harga properti untuk kecamatan Kebayoran Baru, Jakarta Selatan.')

    selected = option_menu(
        menu_title=None,
        options=["Prediksi", "Informasi"],
        icons=["graph-up", "info-square"],
        menu_icon="cast",
        default_index=0,
        orientation="horizontal"
    )

    st.divider()

    if selected == "Prediksi":
        buildingSize = st.text_input('Luas Bangunan (m²)')
        landSize     = st.text_input('Luas Tanah (m²)')
        bedRooms     = st.slider('Jumlah Kamar Tidur', 0, 8, 4)
        bathRooms    = st.slider('Jumlah Kamar Mandi', 0, 8, 4)
        garages      = st.slider('Garasi (Lot Parkir Mobil)', 0, 5, 1)
        landHeight   = st.text_input('Tinggi Tanah (dari permukaan laut dalam dekameter)')
    
        if st.button('Prediksi Harga'):
            price      = 0
            price_pred = predict_price([price, buildingSize, landSize, bedRooms, bathRooms, garages, landHeight])
            st.markdown('#### Harga properti anda')
            st.header(rupiah(price_pred))


    if selected == "Informasi":
        st.subheader('Tentang Model')
        st.markdown('Model prediksi harga properti Kebayoran Baru diciptakan menggunakan algoritma Random Forest Regressor yang memiliki skor akurasi atau R² sebesar 90.2%. Model ini telah di tuning dengan parameter Criterion friedman MSE, max depth 8, min samples split 50, dan number of estimators 100. Model ini dilatih menggunakan data yang terdiri dari 7 atribut dengan 2438 baris data yang diambil dari situs penyedia data open source Kaggle.')
    
        st.subheader('Tentang Kami')
        st.markdown('Kami merupakan empat mahasiswa Universitas Multimedia Nusantara yang terdiri dari Adhelio Reyhandro, Aura Lintang, Fritz Filemon, Sultan Rangga sebagai kelompok 5. Tujuan kami dalam membuat model ini adalah sebagai salah satu syarat kelulusan mata kuliah Data Modelling dengan dosen kami ibu Irmawati, S. Kom., M.M.S.I.')

    st.divider()

if __name__ == '__main__':
    main()