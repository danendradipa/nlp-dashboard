# Sentiment Analysis NLP App (Indonesia)💬
## Project Overview:
This project is an application for sentiment analysis that features two main menus. The first menu provides tools for analyzing sentiment in the Indonesian language. The second menu, Dataset, Cleaning & Analysis, allows users to upload their CSV dataset, select the column to clean, and generate sentiment predictions for each row in the dataset. It also offers additional analyses, such as word cloud, sentiment distribution, most frequent words, and KMeans clustering. Ensure that your dataset contains textual data in the column you select for cleaning.

This project has been deployed on Streamlit and can be accessed through the following link:
https://nlpdashboard-dane.streamlit.app/


## Key Features:
- 🔍 Sentiment Prediction: Users can input text to predict sentiment as positive, negative, or neutral using a trained Logistic Regression model.
- 📊 Dataset Upload & Analysis: Upload your CSV dataset, clean selected columns, generate sentiment predictions for each row, and visualize sentiment distribution, word frequency, word cloud, and perform KMeans clustering to group similar sentiments.
    
## Installation:

1. Clone the repository:
    ```bash
    https://github.com/danendradipa/nlp-dashboard.git
    ```

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## How to run:

1. Run the Streamlit app using the command:

    ```bash
    streamlit run dashboard.py
    ```
