import streamlit as st
import pandas as pd
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="🍛 TastyBot - Indian Recipe Chatbot", layout="centered")

# -------------------- CUSTOM STYLES --------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #f5f3ef, #ddd6c3);
;
        font-family: "Segoe UI", sans-serif;
    }

    /* Center title and make it pop */
    .title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        color: #d35400;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.1);
        margin-bottom: -10px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #555;
        margin-bottom: 25px;
    }

    /* Chat bubbles */
    .user-msg {
        background: #d0f0ff;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        width: fit-content;
        max-width: 80%;
        align-self: flex-end;
        color: #000;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }

    .bot-msg {
        background: #fff3cd;
        padding: 12px 18px;
        border-radius: 18px;
        margin: 8px 0;
        width: fit-content;
        max-width: 85%;
        color: #333;
        box-shadow: 2px 2px 5px rgba(0,0,0,0.1);
    }

    /* Recipe card styling */
    .recipe-card {
        background: white;
        border-radius: 18px;
        padding: 16px;
        margin: 15px 0;
        box-shadow: 0 3px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
    }
    .recipe-card:hover {
        transform: scale(1.02);
        box-shadow: 0 5px 16px rgba(0,0,0,0.25);
    }

    .recipe-title {
        font-size: 22px;
        font-weight: 700;
        color: #d35400;
    }

    .footer {
        text-align: center;
        color: gray;
        margin-top: 25px;
        font-size: 13px;
    }

    .stTextInput > div > div > input {
        border-radius: 16px;
        padding: 12px;
        border: 2px solid #d35400;
    }

    .stButton button {
        background-color: #d35400;
        color: white;
        border-radius: 16px;
        padding: 8px 20px;
        font-weight: 600;
        border: none;
        transition: 0.3s;
    }

    .stButton button:hover {
        background-color: #e67e22;
        transform: scale(1.05);
    }

    </style>
""", unsafe_allow_html=True)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    df = pd.read_csv("IndianFoodDatasetCSV.csv")
    df.dropna(subset=['TranslatedRecipeName', 'TranslatedIngredients', 'TranslatedInstructions'], inplace=True)
    df = df[['TranslatedRecipeName', 'TranslatedIngredients', 'TranslatedInstructions', 'TotalTimeInMins', 'Cuisine', 'Diet']]
    return df

df = load_data()

# -------------------- TF-IDF SETUP --------------------
@st.cache_resource
def create_tfidf_matrix(data):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(data)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = create_tfidf_matrix(df['TranslatedIngredients'])

# -------------------- RECOMMENDATION FUNCTION --------------------
def recommend_recipes(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, tfidf_matrix)
    top_indices = similarity[0].argsort()[-5:][::-1]
    results = df.iloc[top_indices][['TranslatedRecipeName', 'TranslatedIngredients', 'TranslatedInstructions', 'TotalTimeInMins', 'Cuisine', 'Diet']]
    return results

# -------------------- HEADER --------------------
st.markdown("<h1 class='title'>🍽️ TastyBot</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Ask for Indian recipes by ingredients, mood, or cuisine! 🌶️</p>", unsafe_allow_html=True)

# -------------------- CHAT INTERFACE --------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("👩‍🍳 You:", placeholder="E.g. I have paneer and spinach...", key="input_box")

col1, col2 = st.columns([1, 5])
with col1:
    send = st.button("Send")
with col2:
    clear = st.button("Clear Chat")

if send and user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    recs = recommend_recipes(user_input)
    
    if recs.empty:
        bot_response = "😕 Sorry, I couldn’t find anything matching that. Try another ingredient!"
    else:
        bot_response = "✨ Here are some recipes you’ll love!"
    
    st.session_state.chat_history.append({"role": "bot", "content": bot_response})

    for _, row in recs.iterrows():
        st.markdown(f"""
        <div class="recipe-card">
            <div class="recipe-title">🍴 {row['TranslatedRecipeName']}</div>
            <p><b>⏱️ Time:</b> {row['TotalTimeInMins']} mins</p>
            <p><b>🥗 Cuisine:</b> {row['Cuisine']} | <b>Diet:</b> {row['Diet']}</p>
            <p><b>🧂 Ingredients:</b> {row['TranslatedIngredients'][:200]}...</p>
            <details><summary><b>👨‍🍳 Steps</b></summary><p>{row['TranslatedInstructions']}</p></details>
        </div>
        """, unsafe_allow_html=True)

if clear:
    st.session_state.chat_history = []
    st.experimental_rerun()

# -------------------- DISPLAY CHAT --------------------
for chat in st.session_state.chat_history:
    if chat["role"] == "user":
        st.markdown(f"<div class='user-msg'>🧑‍🍳 {chat['content']}</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='bot-msg'>🤖 {chat['content']}</div>", unsafe_allow_html=True)

# -------------------- FOOTER --------------------
st.markdown("<p class='footer'>© 2025 TastyBot |</p>", unsafe_allow_html=True)
