import streamlit as st
import numpy as np
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from textblob import TextBlob
import joblib

kmeans = joblib.load('kmeans_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

st.title('Conversation Topic and Sentiment Analysis')
uploaded_file = st.file_uploader("Choose a JSON/JSONL file", type=["json", "jsonl"])

if uploaded_file is not None:
    file_type = uploaded_file.name.split('.')[-1]
    if file_type == 'json':
        data = json.load(uploaded_file)
    else:
        lines = uploaded_file.readlines()
        data = [json.loads(line) for line in lines]
    conversations = pd.DataFrame(data)
    
    def extract_conversation_text(conversation):
        return ' '.join([turn['value'] for turn in conversation])
    
    conversations['text'] = conversations['conversations'].apply(extract_conversation_text)
    X = vectorizer.transform(conversations['text'])
    conversations['cluster'] = kmeans.predict(X)

    topic_labels = {
        0: 'Physics and Quantum Mechanics',
        1: 'Business and Project Management',
        2: 'Programming and Development',
        3: 'Mathematics and Probability',
        4: 'General Life and Philosophy',
        5: 'Biology and Genetics',
        6: 'Signal Processing and Fourier Analysis',
        7: 'Chemical Processes and Reactions',
        8: 'Neuroscience and Physiology',
        9: 'Algorithms and Problem Solving'
    }
    keywords = {
        'Finance': ['finance', 'investment', 'stock', 'stock market', 'economy', 'bank', 'money', 'financial'],
        'Healthcare': ['healthcare', 'hospital', 'doctor', 'medicine', 'medical', 'clinic', 'health'],
        'Therapy': ['therapy', 'therapist', 'counseling', 'mental health', 'psychology', 'psychiatry'],
         'Content Creation': ['content creation', 'content creator', 'content marketing', 'content strategy', 'content development','instagram', 'youtube', 'tiktok', 'social media'],
        'Gaming': ['gaming', 'game', 'gamer', 'video game', 'esports', 'gaming industry'],
    }

    def assign_predefined_labels(text, keywords):
        for label, words in keywords.items():
            if any(word in text.lower() for word in words):
                return label
        return 'Misc'

    conversations['topic'] = conversations['text'].apply(lambda x: assign_predefined_labels(x, keywords))
    conversations.loc[conversations['topic'] == 'Misc', 'topic'] = conversations['cluster'].map(topic_labels)

    centroids = kmeans.cluster_centers_
    distances = pairwise_distances_argmin_min(X, centroids)[1]
    distance_threshold = 1.0  
    conversations.loc[distances > distance_threshold, 'topic'] = 'Misc'

    st.subheader('Counts of Conversations by Topic')
    topic_counts = conversations['topic'].value_counts()
    st.table(topic_counts)

    def get_sentiment(text):
        analysis = TextBlob(text)
        if analysis.sentiment.polarity > 0:
            return 'positive'
        elif analysis.sentiment.polarity < 0:
            return 'negative'
        else:
            return 'neutral'

    conversations['sentiment'] = conversations['text'].apply(get_sentiment)
    st.subheader('Counts of Conversations by Sentiment')
    sentiment_counts = conversations['sentiment'].value_counts()
    st.table(sentiment_counts)

    st.subheader('Data with Topics and Sentiments')
    conversation_summary = conversations[['id', 'topic', 'sentiment']]
    page_size = 50
    total_pages = len(conversation_summary) // page_size + (len(conversation_summary) % page_size > 0)
    page_number = st.number_input('Page number', min_value=1, max_value=total_pages, value=1)
    
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    paginated_data = conversation_summary.iloc[start_idx:end_idx]

    st.dataframe(paginated_data)
    selected_id = st.selectbox("Select a conversation ID to view details", paginated_data['id'])
