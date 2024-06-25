import streamlit as st
import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import vertexai
from vertexai.generative_models import GenerativeModel

import numpy as np

# Initialize Vertex AI
project_id = "verdant-bus-427506-v7"
vertexai.init(project=project_id, location="us-central1")

# Load the Vertex AI Generative Model
model = GenerativeModel(model_name="gemini-1.5-flash-001")

st.title('Conversation Topic and Sentiment Analysis')
uploaded_file = st.file_uploader("Choose a JSON/JSONL file", type=["json", "jsonl"])

def load_multiple_json_objects(file_content):
    json_objects = file_content.splitlines()
    data = []
    for obj in json_objects:
        obj = obj.strip()
        if obj:
            try:
                data.append(json.loads(obj))
            except json.JSONDecodeError as e:
                st.error(f"Error decoding JSON: {e}")
    return data

def generate_topic_label(documents, keywords):
    prompt = f"""
    Q:
    I have a topic that contains the following documents:
    {documents}

    The topic is described by the following keywords: '{keywords}'.

    Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
    """
    response = model.generate_content(prompt)
    return response.candidates[0].content.parts[0]._raw_part.text.strip()

def get_top_terms_per_cluster(tfidf_matrix, labels, terms, n_terms=10):
    df = pd.DataFrame(tfidf_matrix.todense()).groupby(labels).mean()
    top_terms = {}
    for i, r in df.iterrows():
        top_terms[i] = ', '.join([terms[t] for t in np.argsort(r)[-n_terms:]])
    return top_terms

if uploaded_file is not None:
    file_content = uploaded_file.read().decode('utf-8')
    data = load_multiple_json_objects(file_content)

    extracted_data = []
    for item in data:
        try:
            message_list = json.loads(item[0]['message'])
            userid = item[0]['userid']
            if message_list and isinstance(message_list, list) and len(message_list) > 0:
                message_content = message_list[0]
                query = message_content['query']
                response = message_content['response']
                combined_text = f"Query: {query} Response: {response}"
                extracted_data.append({'userid': userid, 'text': combined_text})
        except json.JSONDecodeError as e:
            continue

    conversations = pd.DataFrame(extracted_data)
    
    # Clustering
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(conversations['text'])
    
    kmeans = KMeans(n_clusters=5, random_state=42)
    conversations['cluster'] = kmeans.fit_predict(X)
    
    # Get top terms for each cluster
    terms = vectorizer.get_feature_names_out()
    top_terms_per_cluster = get_top_terms_per_cluster(X, kmeans.labels_, terms)
    
    # Assigning topic labels
    cluster_groups = conversations.groupby('cluster')['text'].apply(list).reset_index()
    cluster_groups['keywords'] = cluster_groups['cluster'].map(top_terms_per_cluster)
    
    cluster_groups['label'] = cluster_groups.apply(lambda row: generate_topic_label(row['text'], row['keywords']), axis=1)
    
    topic_mapping = cluster_groups.set_index('cluster')['label'].to_dict()
    conversations['topic'] = conversations['cluster'].map(topic_mapping)
    
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
    conversation_summary = conversations[['text', 'topic', 'sentiment']]

    page_size = 50
    total_pages = len(conversation_summary) // page_size + (len(conversation_summary) % page_size > 0)
    page_number = st.number_input('Page number', min_value=1, max_value=total_pages, value=1)

    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    paginated_data = conversation_summary.iloc[start_idx:end_idx]

    st.dataframe(paginated_data)
    selected_id = st.selectbox("Select a conversation userID to view details", paginated_data.index)
    if selected_id:
        selected_conversation = conversations[conversations.index == selected_id]
        st.write("Selected Conversation")
        st.write(selected_conversation[['topic', 'sentiment', 'text']].to_dict(orient='records')[0])
