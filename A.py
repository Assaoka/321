import streamlit as st

# Colocar um site "inteiro" dentro do Streamlit http://127.0.0.1:5500/Backup/Analises/mapa_mortalidade_1991.html

st.markdown("<iframe src='http://127.0.0.1:5500/Backup/Analises/mapa_mortalidade_1991.html' width='100%' height='1000'></iframe>", unsafe_allow_html=True)
