
import streamlit as st
import torch
from transformers import pipeline
import re

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 42px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 10px;
        }

        .subtitle {
            text-align: center;
            color: #666;
            font-size: 18px;
            margin-bottom: 30px;
        }

        .result-box {
            padding: 25px;
            border-radius: 12px;
            margin-top: 20px;
            text-align: center;
        }

        .real-box {
            background-color: #e8f5e9;
            border: 2px solid #4caf50;
        }

        .fake-box {
            background-color: #ffebee;
            border: 2px solid #f44336;
        }

        .warning-box {
            background-color: #fff8e1;
            border: 2px solid #ff9800;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# TITLE
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">📰 AI Fake News Detection</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Analyze news articles using Artificial Intelligence and Natural Language Processing'
    '</div>',
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# LOAD AI MODEL
# ---------------------------------------------------------

@st.cache_resource
def load_model():

    model_name = "hamzab/roberta-fake-news-classification"

    classifier = pipeline(
        "text-classification",
        model=model_name,
        tokenizer=model_name
    )

    return classifier


# ---------------------------------------------------------
# TEXT CLEANING
# ---------------------------------------------------------

def clean_text(text):

    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

with st.sidebar:

    st.header("⚙️ Settings")

    st.write(
        "This application uses a Transformer-based "
        "Natural Language Processing model."
    )

    st.divider()

    st.subheader("How it works")

    st.write("1. Enter a news article.")
    st.write("2. The AI analyzes the text.")
    st.write("3. The model predicts the news category.")
    st.write("4. The prediction confidence is displayed.")

    st.divider()

    st.caption(
        "⚠️ AI predictions are not guaranteed to be factually correct. "
        "Always verify important news using reliable sources."
    )


# ---------------------------------------------------------
# INPUT
# ---------------------------------------------------------

st.subheader("📝 Enter News Article")

news_text = st.text_area(
    "Paste the news article here:",
    height=300,
    placeholder=(
        "Paste the complete news article here..."
    )
)

# ---------------------------------------------------------
# ANALYZE BUTTON
# ---------------------------------------------------------

if st.button("🔍 Analyze News", use_container_width=True):

    if not news_text.strip():

        st.warning("Please enter a news article first.")

    elif len(news_text.strip()) < 50:

        st.warning(
            "Please enter a longer article for a more meaningful analysis."
        )

    else:

        cleaned_text = clean_text(news_text)

        with st.spinner("🤖 AI is analyzing the article..."):

            try:

                classifier = load_model()

                # Transformer models have input-length limits.
                # Analyze the first part of a long article.
                analysis_text = cleaned_text[:4000]

                result = classifier(
                    analysis_text,
                    truncation=True
                )[0]

                label = result["label"]
                confidence = result["score"]

                # -------------------------------------------------
                # INTERPRET MODEL LABEL
                # -------------------------------------------------

                label_upper = label.upper()

                if "FAKE" in label_upper:

                    prediction = "FAKE NEWS"

                elif "REAL" in label_upper:

                    prediction = "REAL NEWS"

                elif label_upper in ["LABEL_0", "0"]:

                    prediction = "FAKE NEWS"

                elif label_upper in ["LABEL_1", "1"]:

                    prediction = "REAL NEWS"

                else:

                    prediction = label

                confidence_percent = confidence * 100

                # -------------------------------------------------
                # DISPLAY RESULT
                # -------------------------------------------------

                st.subheader("📊 Detection Result")

                if prediction == "FAKE NEWS":

                    st.markdown(
                        f"""
                        <div class="result-box fake-box">

                        <h1>🚨 FAKE NEWS</h1>

                        <h3>
                        Confidence: {confidence_percent:.2f}%
                        </h3>

                        <p>
                        The AI model classified this article as
                        potentially misleading or fake.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                elif prediction == "REAL NEWS":

                    st.markdown(
                        f"""
                        <div class="result-box real-box">

                        <h1>✅ REAL NEWS</h1>

                        <h3>
                        Confidence: {confidence_percent:.2f}%
                        </h3>

                        <p>
                        The AI model classified this article as
                        likely genuine.
                        </p>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                else:

                    st.markdown(
                        f"""
                        <div class="result-box warning-box">

                        <h2>Prediction: {prediction}</h2>

                        <h3>
                        Confidence: {confidence_percent:.2f}%
                        </h3>

                        </div>
                        """,
                        unsafe_allow_html=True
                    )

                # -------------------------------------------------
                # CONFIDENCE BAR
                # -------------------------------------------------

                st.write("### Confidence")

                st.progress(float(confidence))

                # -------------------------------------------------
                # MODEL INFORMATION
                # -------------------------------------------------

                with st.expander("ℹ️ Model Information"):

                    st.write(
                        "The application uses a Transformer-based "
                        "text classification model."
                    )

                    st.write(
                        f"Model prediction: `{label}`"
                    )

                    st.write(
                        f"Confidence score: `{confidence:.4f}`"
                    )

                    st.write(
                        f"Characters analyzed: `{len(analysis_text)}`"
                    )

            except Exception as e:

                st.error(
                    "An error occurred while analyzing the article."
                )

                st.exception(e)


# ---------------------------------------------------------
# FOOTER
# ---------------------------------------------------------

st.divider()

st.caption(
    "AI Fake News Detection System • "
    "Built with Python, Streamlit, PyTorch and Transformers"
)