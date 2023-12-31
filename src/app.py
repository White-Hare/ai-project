import streamlit as st
import numpy as np
import pandas as pd

from train import read_and_proccess_data

import matplotlib.pyplot as plt


title = "Yapay Zeka Proje - Mete ARSLAN - 23833301002"

algorithm_text_number_dict = {
    'Odev 3 - KNeighborsClassifier': 3,
    'Odev 4 - GaussianNB': 4,
    'Odev 5.1 - DecisionTreeClassifier': 5,
    'Odev 5.2 - DecisionTreeRegressor': 6,
    'Odev 6 - LogisticRegression': 7,
    'Odev 8 - MLPClassifier': 9,
    'Odev 9 - SVC': 10,
}

default_algorithm = 3
default_iteration_start = 5
default_iteration_end = 10

rerender_key = 'rerender'


def initialize_session_state():
    if rerender_key not in st.session_state:
        st.session_state[rerender_key] = True
       

def page_config():
    st.set_page_config(title, layout='wide')


def side_bar():
    st.sidebar.title("Ayarlar")
    
    algorithm = st.sidebar.selectbox("Algoritma", algorithm_text_number_dict.keys())
    algorithm = algorithm_text_number_dict[algorithm]

    s, e = st.sidebar.slider("Iterasyon Sayisi", 0, 300, (default_iteration_start, default_iteration_end), 1)

    if st.sidebar.button('Calistir'):
        body(algorithm, s, e)
        st.session_state[rerender_key] = False



def header():
    st.title(title)
    st.link_button('Kod - Github', 'https://github.com/White-Hare')

def body(algorithm = 3, iteration_start = default_iteration_start, iteration_end=default_iteration_end):
    (b_df, b_roc), (d_df, d_roc), (w_df, w_roc)  = read_and_proccess_data(algorithm, iteration_start, iteration_end)

    result_section('Siyah', b_df, b_roc)
    result_section('Beraberlik', d_df, d_roc)
    result_section('Beyaz', w_df, w_roc)


    



def result_section(title: str, df: pd.DataFrame, roc: pd.DataFrame):
    st.title(title)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Metrikler")
        st.write(df)

    with col2:
        st.subheader("ROC Curve")
        st.line_chart(roc)





def main():
    initialize_session_state()
    page_config()
    header()
    side_bar()

    if st.session_state[rerender_key]:
        st.session_state[rerender_key] = False
        body()


if __name__ == '__main__':
    main()