import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.express as px
import os
import base64

print('hello')

import kagglehub

st.set_page_config(page_title="Netflix & Amazon Prime", layout="wide")

@st.cache_data
def laad_datasets():
    # Netflix downloaden en inladen
    netflix_map = kagglehub.dataset_download("shivamb/netflix-shows")
    netflix_pad = os.path.join(netflix_map, "netflix_titles.csv")
    df_netflix = pd.read_csv(netflix_pad)

    # Amazon Prime downloaden en inladen
    amazon_map = kagglehub.dataset_download(
        "shivamb/amazon-prime-movies-and-tv-shows"
    )
    amazon_pad = os.path.join(amazon_map, "amazon_prime_titles.csv")
    df_amazon = pd.read_csv(amazon_pad)

    return df_netflix, df_amazon


st.title("Vergelijking Netflix en Amazon Prime")

df_netflix, df_amazon = laad_datasets()

st.subheader("Netflix")
st.dataframe(df_netflix.head())

st.subheader("Amazon Prime")
st.dataframe(df_amazon.head())
