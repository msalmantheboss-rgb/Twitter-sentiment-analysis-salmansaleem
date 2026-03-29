from textblob import TextBlob
import nltk
import pandas as pd
import streamlit as st
import cleantext
import emoji
import os

# Ensure NLTK resources are downloaded
nltk.download('punkt')

st.title("Sentiment Web Analyzer")

# Safe image loading
background_image = 'image.jpg'
if os.path.exists(background_image):
    st.image(background_image, use_column_width=True)
else:
    st.info("Header image 'image.jpg' not found, skipping...")

st.header("Now Scale Your Thoughts")

# ----- Text Analysis -----
with st.expander("Analyze Your Text"):
    text = st.text_input("Text here:", key="manual_text")

    if text:
        blob = TextBlob(text)
        polarity = round(blob.sentiment.polarity, 2)
        st.write('**Polarity:**', polarity)

        if polarity >= 0.1:
            st.write(emoji.emojize("Positive Speech :grinning_face_with_big_eyes:"))
        elif polarity == 0.0:
            st.write(emoji.emojize("Neutral Speech :zipper-mouth_face:"))
        else:
            st.write(emoji.emojize("Negative Speech :disappointed_face:"))

        st.write('**Subjectivity:**', round(blob.sentiment.subjectivity, 2))

    clean_text_input = st.text_input('Clean Your Text:', key="clean_text")
    if clean_text_input:
        cleaned = cleantext.clean(
            clean_text_input,
            fix_unicode=True,
            to_ascii=True,
            lower=True,
            no_line_breaks=True,
            no_urls=True,
            no_emails=True,
            no_phone_numbers=True,
            no_numbers=True,
            no_digits=True,
            no_currency_symbols=True,
            no_punct=True
        )
        st.success(cleaned)

# ----- Excel/CSV Analysis -----
with st.expander('Analyze Excel files'):
    st.write("_**Note**_: Your file must contain the column named 'Tweets' with the text to be analyzed.")
    uploaded_file = st.file_uploader('Upload file', type=['xlsx', 'xls', 'csv'])

    def score(text):
        return TextBlob(str(text)).sentiment.polarity

    def analyze(polarity):
        if polarity >= 0.1:
            return 'Positive'
        elif polarity <= -0.1:
            return 'Negative'
        else:
            return 'Neutral'

    if uploaded_file:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        if 'Tweets' not in df.columns:
            st.error("Column 'Tweets' not found in the file!")
        else:
            df['score'] = df['Tweets'].apply(score)
            df['analysis'] = df['score'].apply(analyze)
            st.write(df.head(10))

            @st.cache_data
            def convert_df(df_input):
                return df_input.to_csv(index=False).encode('utf-8')

            csv = convert_df(df)

            st.download_button(
                label="Download data as CSV",
                data=csv,
                file_name='sentiment.csv',
                mime='text/csv',
            )

# Footer
st.markdown("<br><br><hr style='border: 1px solid black;'>", unsafe_allow_html=True)
st.write("Copy© 2025 Salman Saleem | Made With ❤️ in Pakistan")
