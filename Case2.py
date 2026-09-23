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



# ----------------------------------------------------------

# Datacleaning met pandas, aangepast aan de kolommen in deze datasets.
# Gebruikte functies: drop_duplicates, to_datetime, to_numeric en fillna.
# Bron: https://pandas.pydata.org/docs/
@st.cache_data
def schoon_dataset(df, platform):
    schoon = df.copy()

    # 1. Kolomnamen en tekstwaarden uniform maken.
    schoon.columns = schoon.columns.str.strip()
    for kolom in schoon.select_dtypes(include="object").columns:
        schoon[kolom] = schoon[kolom].str.strip()
        schoon[kolom] = schoon[kolom].replace("", pd.NA)

    # 2. Dubbele titels binnen hetzelfde platform verwijderen.
    # show_id is de unieke identificatiecode in deze datasets.
    if "show_id" in schoon.columns:
        schoon = schoon.drop_duplicates(subset="show_id")
    else:
        schoon = schoon.drop_duplicates()

    # 3. Datatypes corrigeren. Ongeldige waarden worden ontbrekend (NaT/NaN).
    schoon["release_year"] = pd.to_numeric(
        schoon["release_year"], errors="coerce"
    )
    schoon["date_added"] = pd.to_datetime(
        schoon["date_added"], errors="coerce"
    )

    # 4. Een getal uit duration halen, bijvoorbeeld "90 min" wordt 90.
    schoon["duration_value"] = pd.to_numeric(
        schoon["duration"].str.extract(r"(\d+)")[0],
        errors="coerce"
    )

    # 5. Ontbrekende categorische waarden zichtbaar maken.
    # Dit voorkomt dat titels stilletjes uit grafieken verdwijnen.
    for kolom in ["country", "rating", "listed_in", "director"]:
        if kolom in schoon.columns:
            schoon[kolom] = schoon[kolom].fillna("Onbekend")

    # 6. Platform toevoegen, zodat beide datasets later vergeleken kunnen worden.
    schoon["platform"] = platform

    # Zonder titel, type of releasejaar is een vergelijking minder betrouwbaar.
    schoon = schoon.dropna(subset=["title", "type", "release_year"])
    schoon["release_year"] = schoon["release_year"].astype(int)

    return schoon


# Gebaseerd op pandas groupby en explode.
# Bron: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html
@st.cache_data
def genre_telling(df):
    genres = df[["platform", "listed_in"]].copy()
    genres["genre"] = genres["listed_in"].str.split(",")
    genres = genres.explode("genre")
    genres["genre"] = genres["genre"].str.strip()

    return (
        genres.groupby(["platform", "genre"])
        .size()
        .reset_index(name="aantal")
    )


st.title("Vergelijking Netflix en Amazon Prime")
st.caption("Interactieve analyse van films en series uit openbare Kaggle-datasets.")

with st.spinner("Datasets worden geladen en opgeschoond..."):
    df_netflix_raw, df_amazon_raw = laad_datasets()
    df_netflix = schoon_dataset(df_netflix_raw, "Netflix")
    df_amazon = schoon_dataset(df_amazon_raw, "Amazon Prime")

# Bron: https://pandas.pydata.org/docs/reference/api/pandas.concat.html
# Samenvoegen maakt vergelijking tussen platforms mogelijk.
df_alle = pd.concat([df_netflix, df_amazon], ignore_index=True)

# -------- Sidebarfilters --------
st.sidebar.header("Filters")
platform_keuze = st.sidebar.multiselect(
    "Platform",
    options=sorted(df_alle["platform"].unique()),
    default=sorted(df_alle["platform"].unique()),
)
type_keuze = st.sidebar.multiselect(
    "Type content",
    options=sorted(df_alle["type"].unique()),
    default=sorted(df_alle["type"].unique()),
)

min_jaar = int(df_alle["release_year"].min())
max_jaar = int(df_alle["release_year"].max())
jaar_keuze = st.sidebar.slider(
    "Releasejaar",
    min_value=min_jaar,
    max_value=max_jaar,
    value=(min_jaar, max_jaar),
)
zoekterm = st.sidebar.text_input("Zoek op titel")

# Alle filters toepassen op dezelfde dataset.
df_filter = df_alle[
    df_alle["platform"].isin(platform_keuze)
    & df_alle["type"].isin(type_keuze)
    & df_alle["release_year"].between(jaar_keuze[0], jaar_keuze[1])
].copy()

if zoekterm:
    df_filter = df_filter[
        df_filter["title"].str.contains(zoekterm, case=False, na=False)
    ]

# -------- Datakwaliteit en kerncijfers --------
st.header("1. Datakwaliteit")
kolom1, kolom2, kolom3 = st.columns(3)
kolom1.metric("Netflix: rijen vóór opschonen", len(df_netflix_raw))
kolom2.metric("Netflix: rijen na opschonen", len(df_netflix))
kolom3.metric("Amazon Prime: rijen na opschonen", len(df_amazon))

with st.expander("Bekijk ontbrekende waarden na opschonen"):
    ontbrekend = df_alle.isna().sum().sort_values(ascending=False)
    st.dataframe(ontbrekend.rename("aantal_ontbrekend"))

st.header("2. Overzicht van de selectie")
kolom1, kolom2, kolom3 = st.columns(3)
kolom1.metric("Titels", len(df_filter))
kolom2.metric("Films", int((df_filter["type"] == "Movie").sum()))
kolom3.metric("Series", int((df_filter["type"] == "TV Show").sum()))

# -------- Plotly --------
st.header("3. Interactieve Plotly-grafieken")

# Bron: https://plotly.com/python-api-reference/generated/plotly.express.bar.html
# Aantallen worden berekend met pandas groupby.
type_telling = (
    df_filter.groupby(["platform", "type"])
    .size()
    .reset_index(name="aantal")
)

fig_plotly = px.bar(
    type_telling,
    x="platform",
    y="aantal",
    color="type",
    barmode="group",
    text="aantal",
    title="Aantal films en series per platform",
    labels={"platform": "Platform", "type": "Type", "aantal": "Aantal titels"},
)
st.plotly_chart(fig_plotly, use_container_width=True)

jaar_telling = (
    df_filter.groupby(["release_year", "platform"])
    .size()
    .reset_index(name="aantal")
)
fig_lijn = px.line(
    jaar_telling,
    x="release_year",
    y="aantal",
    color="platform",
    markers=True,
    title="Aantal titels per releasejaar",
    labels={"release_year": "Releasejaar", "aantal": "Aantal titels"},
)
st.plotly_chart(fig_lijn, use_container_width=True)

# -------- Matplotlib --------
st.header("4. Matplotlib-grafiek")
st.caption("Histogram: verdeling van releasejaren per platform.")

fig_mpl, ax = plt.subplots(figsize=(11, 5))
for platform in df_filter["platform"].unique():
    waarden = df_filter.loc[
        df_filter["platform"] == platform, "release_year"
    ]
    ax.hist(waarden, bins=20, alpha=0.55, label=platform)

ax.set_title("Verdeling van releasejaren")
ax.set_xlabel("Releasejaar")
ax.set_ylabel("Aantal titels")
ax.legend()
st.pyplot(fig_mpl)
plt.close(fig_mpl)

# -------- Seaborn --------
st.header("5. Seaborn-grafiek")
st.caption("Heatmap: aantal films en series per platform.")

heatmap_data = pd.crosstab(df_filter["platform"], df_filter["type"])
fig_sns, ax = plt.subplots(figsize=(8, 3.5))
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt="d",
    cmap="Blues",
    linewidths=0.5,
    ax=ax,
)
ax.set_title("Films en series per platform")
ax.set_xlabel("Type content")
ax.set_ylabel("Platform")
st.pyplot(fig_sns)
plt.close(fig_sns)

# -------- Genres en tabel --------
st.header("6. Genres en gefilterde tabel")
genres_filter = genre_telling(df_filter)
top_genres = (
    genres_filter.groupby("genre")["aantal"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig_genres = px.bar(
    top_genres,
    x="aantal",
    y="genre",
    orientation="h",
    title="Top 10 genres in de geselecteerde titels",
    labels={"aantal": "Aantal titels", "genre": "Genre"},
)
fig_genres.update_layout(yaxis={"categoryorder": "total ascending"})
st.plotly_chart(fig_genres, use_container_width=True)
# GENRES

fig_countries = px.bar(
    top_genres,
    x="country",
    y="aantal",
    orientation="v",
    title="Top 10 genres in de geselecteerde titels",
    labels={"aantal": "Aantal titels", "genre": "Genre"},
)
fig_countries.update_layout(yaxis={"countryorder": "total ascending"})
st.plotly_chart(fig_genres, use_container_width=True)


st.dataframe(
    df_filter.sort_values("release_year", ascending=False),
    use_container_width=True,
    hide_index=True,
)

csv = df_filter.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download gefilterde data als CSV",
    data=csv,
    file_name="netflix_amazon_selectie.csv",
    mime="text/csv",
)

# -------- Productielanden --------
st.subheader("Top 10 productielanden")

# Maak een aparte tabel voor landen.
# Een titel met bijvoorbeeld "United States, India" telt voor beide landen mee.
landen = df_filter[["country", "platform"]].copy()

landen["country"] = landen["country"].str.split(",")
landen = landen.explode("country")
landen["country"] = landen["country"].str.strip()

# Tel het aantal titels per land en platform.
landen_telling = (
    landen.groupby(["country", "platform"])
    .size()
    .reset_index(name="aantal")
)

# Bepaal de tien landen met de meeste titels in totaal.
top_land_namen = (
    landen_telling.groupby("country")["aantal"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .index
)

# Houd alleen deze top 10 over.
top_landen = landen_telling[
    landen_telling["country"].isin(top_land_namen)
]

# Maak de grafiek.
fig_countries = px.bar(
    top_landen,
    x="country",
    y="aantal",
    color="platform",
    barmode="group",
    title="Top 10 productielanden per platform",
    labels={
        "country": "Productieland",
        "aantal": "Aantal titels",
        "platform": "Platform",
    },
    category_orders={"country": list(top_land_namen)},
)

st.plotly_chart(fig_countries, use_container_width=True)


