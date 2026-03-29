from textblob import TextBlob
import nltk
import pandas as pd
import streamlit as st
import cleantext
import emoji

# Ensure NLTK resources are available
nltk.download('punkt')

# Page Configuration
st.set_page_config(page_title="Sentiment Web Analyzer", layout="centered")

st.title("Sentiment Web Analyzer")

# Handling the background image safely
try:
    background_image = '"image.jpg"'
    st.image(background_image, use_container_width=True)
except:
    st.warning("Update 'image.jpg' in your directory to display the header image.")

st.header("Now Scale Your Thoughts")

# ----- Text Analysis Section -----
with st.expander("Analyze Your Text"):
    text = st.text_input("Text here:", key="sentiment_input")

    if text:
        blob = TextBlob(text)
        # Use LaTeX for the Polarity calculation if you want to show the math,
        # but for simple display, standard text is cleaner.
        polarity = round(blob.sentiment.polarity, 2)
        st.write(f'**Polarity:** {polarity}')

        if polarity >= 0.1:
            st.write(emoji.emojize("Positive Speech :grinning_face_with_big_eyes:"))
        elif polarity == 0.0:
            st.write(emoji.emojize("Neutral Speech :zipper-mouth_face:"))
        else:
            st.write(emoji.emojize("Negative Speech :disappointed_face:"))

        st.write(f'**Subjectivity:** {round(blob.sentiment.subjectivity, 2)}')

    st.markdown("---")
    clean_text_input = st.text_input('Clean Your Text:', key="clean_input")
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
        st.success(f"Cleaned Text: {cleaned}")

# ----- Excel/CSV Analysis Section -----
with st.expander('Analyze Excel or CSV files'):
    st.info("Note: Your file must contain a column named **'Tweets'**.")
    uploaded_file = st.file_uploader('Upload file', type=['xlsx', 'xls', 'csv'])


    def get_score(text):
        return TextBlob(str(text)).sentiment.polarity


    def get_analysis(polarity):
        if polarity >= 0.1:
            return 'Positive'
        elif polarity <= -0.1:
            return 'Negative'
        else:
            return 'Neutral'


    if uploaded_file:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            if 'Tweets' not in df.columns:
                st.error("Column 'Tweets' not found! Please check your file headers.")
            else:
                with st.spinner('Analyzing sentiments...'):
                    df['score'] = df['Tweets'].apply(get_score)
                    df['analysis'] = df['score'].apply(get_analysis)

                st.write("### Analysis Preview")
                st.dataframe(df.head(10))


                @st.cache_data
                def convert_df(df_to_save):
                    return df_to_save.to_csv(index=False).encode('utf-8')


                csv_data = convert_df(df)

                st.download_button(
                    label="Download Full Results as CSV",
                    data=csv_data,
                    file_name='sentiment_results.csv',
                    mime='text/csv',
                )
        except Exception as e:
            st.error(f"An error occurred while processing the file: {e}")

# Footer
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown("<hr style='border: 1px solid #bbb;'>", unsafe_allow_html=True)
st.caption("Copy© 2025 Salman Saleem | Made With ❤️ in Pakistan")
