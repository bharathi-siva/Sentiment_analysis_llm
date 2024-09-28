from flask import Flask, request, jsonify, render_template
import pandas as pd
import os
from groq import Groq

app = Flask(__name__)

# Set up Groq API key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if GROQ_API_KEY is None:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

client = Groq(api_key=GROQ_API_KEY)

# Function to analyze sentiment using Groq API
def analyze_sentiment(text):
    try:
        response = client.chat.completions.create(
            messages=[{"role": "user", "content": text}],
            model="llama3-8b-8192"  # Example model, replace with Groq's sentiment model if available
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing sentiment: {str(e)}"

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/analyze', methods=['POST'])
def analyze_reviews():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in request"}), 400

    file = request.files['file']

    # Check for valid file formats
    if not (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
        return jsonify({"error": "Invalid file format. Please upload CSV or XLSX file."}), 400

    try:
        # Read file based on its format
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.filename.endswith('.xlsx'):
            df = pd.read_excel(file)

        # Check for the 'review' column
        if 'review' not in df.columns:
            return jsonify({"error": "Missing 'review' column in the file"}), 400

        reviews = df['review'].dropna().tolist()

        # Sentiment scores initialization
        sentiments = {"positive": 0, "negative": 0, "neutral": 0}

        # Analyze each review's sentiment
        for review in reviews:
            sentiment = analyze_sentiment(review)
            
            # Update sentiment scores based on the result
            if "positive" in sentiment.lower():
                sentiments["positive"] += 1
            elif "negative" in sentiment.lower():
                sentiments["negative"] += 1
            else:
                sentiments["neutral"] += 1

        # Return sentiment scores as JSON
        return jsonify(sentiments)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
