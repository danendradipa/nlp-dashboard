import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import re

# Load the trained pipeline
pipeline = joblib.load('model/model_lr.joblib')

# Set page config
st.set_page_config(
    page_title="Sentiment Analysis Tool", 
    page_icon="📊",  
)

# Cleaning function
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'http\S+', '', string)  # Remove links
    string = re.sub(r'@\w+', '', string)     # Remove mentions
    string = re.sub(r'#\w+', '', string)     # Remove hashtags
    string = re.sub(r'\d+', '', string)      # Remove numbers
    string = re.sub(r'[^\w\s]', '', string)  # Remove special characters
    return string

# Main app function
def main():
    st.sidebar.title("Sentiment Analysis App")
    menu = ["Sentiment Tools", "Dataset, Cleaning & Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Menu 1: Sentiment Tools
    if choice == "Sentiment Tools":
        st.title("Sentiment Analysis Tools")
        st.write("""
        This tool allows you to input any text and quickly analyze its sentiment. 
        With just a few clicks, you can understand whether the sentiment is positive, negative, or neutral.
        It’s a simple way to gauge the emotional tone behind your text.
        """)

        with st.form("nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        if submit_button:
            sentiment = pipeline.predict([raw_text])[0]
            st.write(f"Predicted Sentiment: {sentiment}")

            if sentiment == "positive":
                st.markdown("# Positive 😃")
            elif sentiment == "negative":
                st.markdown("# Negative 😡")
            else:
                st.markdown("# Neutral 😐")
    
    # Menu 2: Dataset, Cleaning & Analysis
    elif choice == "Dataset, Cleaning & Analysis":
        st.title("Dataset Upload & Analysis")
        st.write("""
        Upload your dataset (CSV) for dynamic sentiment analysis. The dataset must have a `full_text` column 
        for the analysis to work properly.
        """)

        # File uploader
        uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)

            # Cleaning the 'full_text' column
            if 'full_text' in data.columns:
                data['cleaned_text'] = data['full_text'].apply(cleansing)

                # Predict sentiment
                data['sentiment'] = pipeline.predict(data['cleaned_text'])

                # Display dataset
                st.subheader("Cleaned Dataset with Sentiment Prediction")
                st.write("This is the cleaned dataset with a new column for sentiment predictions.")
                st.write(data)

                # Sentiment Distribution
                st.subheader("Sentiment Distribution")
                st.write("Visualize the distribution of sentiment (positive, negative, and neutral) within your dataset.")
                sentiment_counts = data['sentiment'].value_counts()
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
                st.pyplot(fig)

                # Word Cloud
                st.subheader("Word Cloud")
                st.write("This word cloud shows the most frequent words used in the dataset after cleaning.")
                st.subheader("Word Cloud")
                text_data = ' '.join(data['cleaned_text'].astype(str).tolist())
                wc = WordCloud(background_color='black', max_words=500, width=800, height=400).generate(text_data)
                plt.figure(figsize=(10, 10))
                plt.imshow(wc, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)

                # Bar chart of most frequent words
                st.subheader("Most Frequent Words")
                st.write("This bar chart displays the 12 most common words in your dataset and their frequency.")
                words = text_data.split()
                word_counts = Counter(words)
                top_words = word_counts.most_common(12)
                words, counts = zip(*top_words)
                fig, ax = plt.subplots()
                ax.bar(words, counts)
                ax.set_xticklabels(words, rotation=45)
                st.pyplot(fig)

                # KMeans Clustering
                st.subheader("KMeans Clustering")
                st.write("This section shows clustering results based on the text data after vectorization.")
                vectorizer = TfidfVectorizer()
                X = vectorizer.fit_transform(data['cleaned_text'])
                true_k = 8
                model = KMeans(n_clusters=true_k)
                model.fit(X)

                order_centroids = model.cluster_centers_.argsort()[:, ::-1]
                terms = vectorizer.get_feature_names_out()
                for i in range(true_k):
                    st.write(f"Cluster {i}: {', '.join(terms[ind] for ind in order_centroids[i, :10])}")

                # Silhouette score
                score = silhouette_score(X, model.labels_)
                st.write(f"Silhouette Score: {score:.2f}")

                # PCA for visualization
                pca = PCA(n_components=2)
                reduced_features = pca.fit_transform(X.toarray())
                reduced_cluster_centers = pca.transform(model.cluster_centers_)
                plt.figure(figsize=(10, 6))
                scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=model.labels_, cmap='viridis', alpha=0.6)
                plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='red', label='Centroids')
                plt.title('KMeans Clustering Visualization')
                plt.xlabel('PCA Component 1')
                plt.ylabel('PCA Component 2')
                plt.colorbar(scatter)
                plt.legend()
                st.pyplot(plt)

if __name__ == '__main__':
    main()
