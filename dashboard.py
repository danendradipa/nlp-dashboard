import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import re

# Load the trained pipeline
pipeline = joblib.load('model/model_lr.joblib')

# Config page
st.set_page_config(
    page_title="Sentiment Analysis Tool", 
    page_icon="📊",  
)

# Fungsi cleaning
def cleansing(sent):
    string = sent.lower()
    string = re.sub(r'http\S+', '', string)  # Menghapus link
    string = re.sub(r'@\w+', '', string)     # Menghapus mention
    string = re.sub(r'#\w+', '', string)     # Menghapus hashtag
    string = re.sub(r'\d+', '', string)      # Menghapus angka
    string = re.sub(r'[^\w\s]', '', string)  # Menghapus karakter khusus
    return string

def main():
    st.title("Sentiment Analysis NLP App (Indonesia) 💬")

    # Add sidebar with 3 menu options
    menu = ["Sentiment Tools", "Dataset, Cleaning & Analysis", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Sentiment Tools":
        st.subheader("Sentiment Analysis Tools")
        st.write("""
        This tool allows you to input indonesian text and quickly analyze its sentiment. 
        With just a few clicks, you can understand whether the sentiment is positive, negative, or neutral.
        It’s a simple way to gauge the emotional tone behind your text.
        """)
        with st.form("nlpForm"):
            raw_text = st.text_area("Enter Text Here")
            submit_button = st.form_submit_button(label='Analyze')

        # Layout
        col1, col2 = st.columns(2)
        if submit_button:
            with col1:
                st.info("Results")
                # Predict sentiment of the entire text using the pipeline
                sentiment = pipeline.predict([raw_text])[0]
                st.write(f"Predicted Sentiment: {sentiment}")

                # Emoji based on prediction
                if sentiment == "positive":
                    st.markdown("# Positive 😃")
                elif sentiment == "negative":
                    st.markdown("# Negative 😡")
                else:
                    st.markdown("# Neutral 😐")

            with col2:
                st.info("Token Sentiment")
                # Analyze sentiment for individual tokens
                token_sentiments = analyze_token_sentiment(raw_text, pipeline)
                st.write(token_sentiments)

    elif choice == "Dataset, Cleaning & Analysis":
        st.subheader("Dataset, Cleaning & Analysis")
        st.write("""
        Upload your CSV dataset, select the column to clean, and generate sentiment predictions. 
        Ensure your dataset contains textual data in the column you select for cleaning.
        """)

        uploaded_file = st.file_uploader("Upload a CSV file", type="csv")
        
        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("### Dataset Preview:")
            st.dataframe(data.head())

            # Select the column for cleaning
            column_to_clean = st.selectbox("Select the column to clean", data.columns)
            
            if st.button("Clean and Analyze"):
                st.write(f"Cleaning text in column: {column_to_clean}")

                # Apply the cleansing function
                data['cleaned_text'] = data[column_to_clean].apply(cleansing)

                # Display dataset with cleaned column
                st.write("### Dataset with Cleaned Text:")
                st.write("""
                This section shows the dataset after the selected text column has been cleaned. 
                The cleaning process includes removing links, mentions, hashtags, numbers, and special characters, 
                making the text more suitable for analysis.
                """)
                st.dataframe(data[[column_to_clean, 'cleaned_text']])

                st.markdown("---")

                # Predict sentiment
                data['sentiment'] = data['cleaned_text'].apply(lambda x: pipeline.predict([x])[0])
                st.write("### Dataset with Sentiment Prediction:")
                st.write("""
                This section displays the dataset with a new 'sentiment' column that contains the predicted sentiment for each text entry. 
                The predictions are based on the cleaned text data, providing insights into the sentiment trends across the dataset.
                """)
                st.dataframe(data[['cleaned_text', 'sentiment']])
                
                st.markdown("---")
                
                # Sentiment Distribution
                sentiment_counts = data['sentiment'].value_counts()
                st.subheader("Sentiment Distribution")
                st.write("This chart shows the distribution of predicted sentiment labels across the dataset.")

                # Customize bar colors based on sentiment
                colors = ['blue' if sentiment == 'neutral' else 'green' if sentiment == 'positive' else 'red' for sentiment in sentiment_counts.index]

                # Plot the sentiment distribution with custom colors
                fig, ax = plt.subplots()
                sentiment_counts.plot(kind='bar', color=colors, ax=ax)

                # Add labels and title
                ax.set_title('Sentiment Distribution')
                ax.set_xlabel('Sentiment')
                ax.set_ylabel('Count')

                # Display the chart in Streamlit
                st.pyplot(fig)
                # Display the number of positive, negative, and neutral sentiments
                st.write("#### Sentiment Count")
                st.write("""
                    This section displays the count of predicted sentiments (positive, negative, and neutral) across the dataset.
                    It provides an overview of how opinions are distributed within the data.
                """)
                st.write(sentiment_counts)

                st.markdown("---")

                # Word Cloud
                st.subheader("Word Cloud")
                st.write("This word cloud visualizes the most frequent words in the cleaned text data.")                
                generate_wordcloud(data)

                st.markdown("---")

                # Bar Chart of Frequent Words
                st.subheader("Frequent Words")
                st.write("This bar chart highlights the top 12 most frequent words in the cleaned text data.")
                plot_top_words(data)

                st.markdown("---")

                # KMeans Clustering
                st.subheader("KMeans Clustering")
                st.write("This section displays the results of clustering the cleaned text data using KMeans. Each cluster is represented by key terms.")

                if 'cleaned_text' in data.columns:
                    # Gunakan kolom 'cleaned_text' untuk clustering
                    text_data = data['cleaned_text'].to_list()

                    # Vectorization
                    vectorizer = TfidfVectorizer()
                    X = vectorizer.fit_transform(text_data)

                    # KMeans clustering
                    true_k = 8
                    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1, random_state=42)
                    model.fit(X)

                    # Display clusters
                    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
                    terms = vectorizer.get_feature_names_out()
                    cluster_info = ""
                    for i in range(true_k):
                        cluster_info += f"Cluster {i}: " + ', '.join([terms[ind] for ind in order_centroids[i, :10]]) + "\n"
                    st.write("### Cluster Terms")
                    st.text(cluster_info)

                    # Silhouette Score
                    from sklearn.metrics import silhouette_score
                    score = silhouette_score(X, model.labels_)
                    st.write(f"### Silhouette Score: {score:.2f}")

                    # PCA for 2D visualization
                    pca = PCA(n_components=2, random_state=0)
                    reduced_features = pca.fit_transform(X.toarray())
                    reduced_cluster_centers = pca.transform(model.cluster_centers_)

                    # Plotting the clusters
                    plt.figure(figsize=(10, 6))
                    scatter = plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=model.labels_, cmap='viridis', alpha=0.6)
                    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='red', label='Centroids')
                    plt.title('KMeans Clustering Visualization')
                    plt.xlabel('PCA Component 1')
                    plt.ylabel('PCA Component 2')
                    plt.colorbar(scatter)
                    plt.legend()
                    st.pyplot(plt)
                else:
                    st.write("Kolom 'cleaned_text' tidak ditemukan. Silakan lakukan cleaning terlebih dahulu.")




    else:
        st.subheader("About")
        st.write("This is a simple sentiment analysis app built using Streamlit.")

# Word Cloud function
def generate_wordcloud(data):
    # Combine the full_text column into one large string
    data_text = ' '.join(data['cleaned_text'].astype(str).tolist())

    # Generate word cloud
    wc = WordCloud(background_color='black', max_words=500, width=800, height=400).generate(data_text)

    # Display the word cloud
    plt.figure(figsize=(10, 10))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

# Function to plot top frequent words
def plot_top_words(data):
    text = ' '.join(data['cleaned_text'].astype(str).tolist())
    words = text.split()
    word_counts = Counter(words)
    top_words = word_counts.most_common(12)
    words, counts = zip(*top_words)

    colors = plt.cm.Paired(range(len(words)))
    plt.figure(figsize=(10, 6))
    bars = plt.bar(words, counts, color=colors)
    plt.xlabel('Kata')
    plt.ylabel('Frekuensi')
    plt.title('Kata yang sering muncul')
    plt.xticks(rotation=45)
    
    for bar, num in zip(bars, counts):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, num + 1, str(num), fontsize=12, color='black', ha='center')

    st.pyplot(plt)

def analyze_token_sentiment(docx, pipeline):
    pos_list = []
    neg_list = []
    neu_list = []

    for word in docx.split():
        # Predict sentiment for each token using the trained pipeline
        prediction = pipeline.predict([word])[0]

        # Check if the pipeline supports predict_proba
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba([word])
            proba_max = proba.max()  # Get the highest probability
        else:
            proba_max = None  # If predict_proba is not available, return None

        if prediction == 'positive':
            pos_list.append([word, proba_max])
        elif prediction == 'negative':
            neg_list.append([word, proba_max])
        else:
            neu_list.append(word)
    
    result = {
        'positives': pos_list,
        'negatives': neg_list,
        'neutral': neu_list,
    }
    return result

if __name__ == '__main__':
    main()
