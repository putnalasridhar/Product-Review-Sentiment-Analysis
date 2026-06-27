import streamlit as st
import tensorflow as tf
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ================= Load Model =================
@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model("sentiment_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_assets()

MAXLEN = 100   # Max length used during training time

# ================= Page Config =================
st.set_page_config(
    page_title="Sentiment Analysis",
    layout="centered"
)

# ================= CSS =================
st.markdown("""
<style>
.stApp{
    background-color:#000000;
    color:white;
}
h1,h2,h3,p,label{
    color:white !important;
}
textarea{
    background-color:#111111 !important;
    color:white !important;
    border:2px solid #00BFFF !important;
    border-radius:10px !important;
}
.stButton>button{
    background:#00BFFF;
    color:white;
    border-radius:10px;
    border:none;
    font-size:18px;
    height:50px;
    width:100%;
}
.stButton>button:hover{
    background:#0099cc;
}
.result{
    padding:15px;
    border-radius:10px;
    font-size:22px;
    text-align:center;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

# ================= Title =================
st.title("Sentiment Analysis App")
st.write("Enter a review and click **Predict**.")

# ================= Input =================
review = st.text_area("Movie Review")

# ================= Prediction =================
if st.button("Predict Sentiment"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:
        # Preprocessing: Convert text to lowercase and remove trailing spaces
        cleaned_review = review.lower().strip()
        
        # Tokenization & Padding
        seq = tokenizer.texts_to_sequences([cleaned_review])
        
        # Note: If padding='post' was used during training, use padding='post' here as well
        padded = pad_sequences(seq, maxlen=MAXLEN, padding='pre')

        # Predict
        prediction = model.predict(padded, verbose=0)[0][0]

        # UI Response (If 1 = Positive, 0 = Negative)
        if prediction >= 0.6:
            st.markdown(
                "<div class='result' style='background:#006400;'>Positive Review</div>",
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                "<div class='result' style='background:#8B0000;'>Negative Review</div>",
                unsafe_allow_html=True
            )

        st.write(f"Prediction Score : **{prediction:.4f}**")