

from flask import Flask, render_template, request, redirect, url_for
import json
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from textblob import TextBlob
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def load_conversations(file_path):
    processed_data = []
    with open(file_path, 'r') as f:
        for line in f:
            conversation = json.loads(line)
            human_messages = [msg['value'] for msg in conversation['conversations'] if msg['from'] == 'human']
            processed_data.append(' '.join(human_messages))
    return processed_data

def cluster_conversations(conversations, n_clusters=10):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(conversations)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(tfidf_matrix)
    
    topics = []
    for i in range(n_clusters):
        cluster_docs = [conversations[j] for j in range(len(conversations)) if cluster_labels[j] == i]
        if cluster_docs:
            tfidf = vectorizer.transform(cluster_docs)
            top_word_indices = np.asarray(tfidf.sum(axis=0)).flatten().argsort()[-5:]
            top_words = [vectorizer.get_feature_names_out()[i] for i in top_word_indices]
            topics.append(' '.join(top_words))
        else:
            topics.append('Misc')
    
    return [topics[label] for label in cluster_labels]

def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0.1:
        return 'positive'
    elif polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

def get_sentiments(conversations):
    return [analyze_sentiment(conv) for conv in conversations]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(url_for('index'))
    
    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    if file and (file.filename.endswith('.json') or file.filename.endswith('.jsonl')):
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # Process the uploaded file
        conversations = load_conversations(file_path)
        topics = cluster_conversations(conversations)
        sentiments = get_sentiments(conversations)
        
        data = pd.DataFrame({
            'Conversation No': range(1, len(conversations) + 1),
            'Topic': topics,
            'Sentiment': sentiments
        })

        # Calculate Counts
        topic_counts = data['Topic'].value_counts().reset_index()
        topic_counts.columns = ['Topic', 'Count']
        sentiment_counts = data['Sentiment'].value_counts().reset_index()
        sentiment_counts.columns = ['Sentiment', 'Count']

        # Store data in session for use in sessions page
        global session_data
        session_data = data

        return render_template(
            'index.html', 
            topic_counts=topic_counts.to_dict(orient='records'),
            sentiment_counts=sentiment_counts.to_dict(orient='records'),
            conversations=data[['Conversation No', 'Topic', 'Sentiment']].to_dict(orient='records')
        )
    else:
        return redirect(url_for('index'))

@app.route('/sessions')
def sessions():
    global session_data
    if session_data is not None:
        conversations = session_data[['Conversation No', 'Topic', 'Sentiment']].to_dict(orient='records')
        return render_template('sessions.html', conversations=conversations)
    else:
        return redirect(url_for('index'))

if __name__ == '__main__':
    session_data = None
    app.run(debug=True)
