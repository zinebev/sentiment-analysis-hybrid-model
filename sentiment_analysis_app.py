# Complete Sentiment Analysis Application
# This includes data preprocessing, model training, and Streamlit interface

import os
import time
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
import re
from scipy.special import softmax

warnings.filterwarnings('ignore')

class SentimentAnalyzer:
    def __init__(self):
        self.logreg_model = None
        self.vectorizer = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.emotions = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'love']
        
    def preprocess_text(self, text):
        """Basic text preprocessing"""
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'[@#](\w+)', r'\1', text)
        
        # Remove extra punctuation (keep some for context)
        text = re.sub(r'[^\w\s!?.,]', ' ', text)
        
        # Handle repeated characters
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_csv_data(self, uploaded_file):
        """Load data from uploaded CSV file"""
        try:
            # Read the CSV file
            df = pd.read_csv(uploaded_file)
            
            # Check if the required columns exist
            if 'text' not in df.columns or 'label' not in df.columns:
                st.error("CSV file must have 'text' and 'label' columns")
                return None, None
            
            # Clean and prepare data
            texts = []
            labels = []
            
            for _, row in df.iterrows():
                text = self.preprocess_text(str(row['text']))
                if text and pd.notna(row['label']):  # Only add non-empty texts with valid labels
                    texts.append(text)
                    labels.append(str(row['label']).lower())
            
            if len(texts) == 0:
                st.error("No valid data found in the CSV file!")
                return None, None
            
            st.success(f"Loaded {len(texts)} samples from CSV")
            return texts, labels
            
        except Exception as e:
            st.error(f"Error loading CSV file: {str(e)}")
            return None, None
    
    def load_or_train_logreg(self, uploaded_file=None, force_retrain=False):
        """Load or train Logistic Regression model"""
        model_path = './models/logreg_model.joblib'
        vectorizer_path = './models/tfidf_vectorizer.joblib'
        
        # Create models directory if it doesn't exist
        os.makedirs('./models', exist_ok=True)
        
        if os.path.exists(model_path) and os.path.exists(vectorizer_path) and not force_retrain:
            st.info("Loading saved Logistic Regression model...")
            self.vectorizer = joblib.load(vectorizer_path)
            self.logreg_model = joblib.load(model_path)
            return True
        else:
            if uploaded_file is not None:
                st.info("Training new Logistic Regression model...")
                return self.train_logreg_model(uploaded_file, model_path, vectorizer_path)
            else:
                st.warning("No training data provided and no saved model found.")
                return False
    
    def train_logreg_model(self, uploaded_file, model_path, vectorizer_path):
        """Train Logistic Regression model"""
        try:
            # Load data from uploaded CSV
            texts, labels = self.load_csv_data(uploaded_file)
            if texts is None or labels is None:
                return False
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42, stratify=labels)
            
            # TF-IDF Vectorization
            self.vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            X_train_tfidf = self.vectorizer.fit_transform(X_train)
            
            # Train Logistic Regression
            self.logreg_model = LogisticRegression(
                max_iter=1000,
                random_state=42,
                class_weight='balanced'
            )
            
            start_time = time.time()
            self.logreg_model.fit(X_train_tfidf, y_train)
            training_time = time.time() - start_time
            
            # Evaluate
            X_test_tfidf = self.vectorizer.transform(X_test)
            y_pred = self.logreg_model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            st.success(f"Training completed in {training_time:.2f} seconds")
            st.success(f"Model accuracy: {accuracy:.4f}")
            
            # Save models
            joblib.dump(self.logreg_model, model_path)
            joblib.dump(self.vectorizer, vectorizer_path)
            st.success("Models saved successfully!")
            
            return True
            
        except Exception as e:
            st.error(f"Error training model: {str(e)}")
            return False
    
    def load_distilbert(self):
        """Load DistilBERT model"""
        try:
            model_name = "bhadresh-savani/distilbert-base-uncased-emotion"
            
            with st.spinner("Loading DistilBERT model..."):
                self.distilbert_tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(model_name)
                
                # Set to evaluation mode
                self.distilbert_model.eval()
                
            st.success("DistilBERT model loaded successfully!")
            return True
            
        except Exception as e:
            st.error(f"Error loading DistilBERT: {str(e)}")
            return False
    
    def predict_logreg(self, text):
        """Predict using Logistic Regression"""
        try:
            if self.logreg_model is None or self.vectorizer is None:
                return {"error": "Logistic Regression model not loaded"}
            
            # Preprocess and vectorize
            clean_text = self.preprocess_text(text)
            if not clean_text:
                return {"error": "Text became empty after preprocessing"}
            
            text_tfidf = self.vectorizer.transform([clean_text])
            
            # Predict
            prediction = self.logreg_model.predict(text_tfidf)[0]
            probabilities = self.logreg_model.predict_proba(text_tfidf)[0]
            
            # Create results
            prob_dict = dict(zip(self.logreg_model.classes_, probabilities))
            confidence = max(probabilities)
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': {k: float(v) for k, v in prob_dict.items()}
            }
            
        except Exception as e:
            return {"error": f"Prediction error: {str(e)}"}
    
    def predict_distilbert(self, text):
        """Predict using DistilBERT"""
        try:
            if self.distilbert_model is None or self.distilbert_tokenizer is None:
                return {"error": "DistilBERT model not loaded"}
            
            # Tokenize
            inputs = self.distilbert_tokenizer(
                text, 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.distilbert_model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predictions = predictions.cpu().numpy()[0]
            
            # Get emotion labels (DistilBERT model specific)
            emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
            
            # Create results
            prob_dict = dict(zip(emotion_labels, predictions))
            prediction = emotion_labels[np.argmax(predictions)]
            confidence = float(max(predictions))
            
            return {
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': {k: float(v) for k, v in prob_dict.items()}
            }
            
        except Exception as e:
            return {"error": f"DistilBERT prediction error: {str(e)}"}
    
    def predict_hybrid(self, text, logreg_weight=0.5, distilbert_weight=0.5):
        """Predict using hybrid approach"""
        try:
            # Get predictions from both models
            logreg_result = self.predict_logreg(text)
            distilbert_result = self.predict_distilbert(text)
            
            if 'error' in logreg_result:
                return logreg_result
            if 'error' in distilbert_result:
                return distilbert_result
            
            # Get all unique emotions
            all_emotions = set(logreg_result['probabilities'].keys()) | set(distilbert_result['probabilities'].keys())
            
            # Calculate weighted average probabilities
            hybrid_probs = {}
            for emotion in all_emotions:
                logreg_prob = logreg_result['probabilities'].get(emotion, 0.0)
                distilbert_prob = distilbert_result['probabilities'].get(emotion, 0.0)
                
                hybrid_probs[emotion] = (logreg_weight * logreg_prob + 
                                       distilbert_weight * distilbert_prob)
            
            # Get final prediction
            prediction = max(hybrid_probs, key=hybrid_probs.get)
            confidence = hybrid_probs[prediction]
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'probabilities': hybrid_probs,
                'logreg_result': logreg_result,
                'distilbert_result': distilbert_result
            }
            
        except Exception as e:
            return {"error": f"Hybrid prediction error: {str(e)}"}

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Multi-Model Sentiment Analysis",
        page_icon="🎭",
        layout="wide"
    )
    
    st.title("🎭 Multi-Model Sentiment Analysis")
    st.markdown("Compare three different approaches to sentiment analysis!")
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SentimentAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for model loading
    st.sidebar.header("⚙️ Model Configuration")
    
    # File upload for training data
    uploaded_file = st.sidebar.file_uploader(
        "Upload Training Dataset (CSV)", 
        type=['csv'],
        help="Upload a CSV file with 'text' and 'label' columns"
    )
    
    # Load Logistic Regression model
    if st.sidebar.button("🔄 Load/Train Logistic Regression"):
        analyzer.load_or_train_logreg(uploaded_file)
    
    # Load DistilBERT model
    if st.sidebar.button("🤖 Load DistilBERT"):
        analyzer.load_distilbert()
    
    # Force retrain option
    if st.sidebar.button("🔥 Force Retrain LogReg"):
        if uploaded_file is not None:
            analyzer.load_or_train_logreg(uploaded_file, force_retrain=True)
        else:
            st.sidebar.error("Please upload a training dataset first!")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("📝 Text Input")
        
        # Text input
        user_text = st.text_area(
            "Enter your text for sentiment analysis:",
            height=150,
            placeholder="Type or paste your text here..."
        )
        
        # Model selection
        model_choice = st.selectbox(
            "Choose Model:",
            ["Logistic Regression", "DistilBERT", "Hybrid (Combined)"]
        )
        
        # Hybrid model weights (only show if hybrid is selected)
        if model_choice == "Hybrid (Combined)":
            st.subheader("⚖️ Model Weights")
            logreg_weight = st.slider("Logistic Regression Weight", 0.0, 1.0, 0.5, 0.1)
            distilbert_weight = 1.0 - logreg_weight
            st.write(f"DistilBERT Weight: {distilbert_weight:.1f}")
        
        # Predict button
        if st.button("🔮 Predict Sentiment", type="primary"):
            if not user_text.strip():
                st.warning("Please enter some text!")
            else:
                with st.spinner("Analyzing sentiment..."):
                    if model_choice == "Logistic Regression":
                        result = analyzer.predict_logreg(user_text)
                    elif model_choice == "DistilBERT":
                        result = analyzer.predict_distilbert(user_text)
                    else:  # Hybrid
                        result = analyzer.predict_hybrid(user_text, logreg_weight, distilbert_weight)
                    
                    # Display results
                    if 'error' in result:
                        st.error(result['error'])
                    else:
                        display_results(result, model_choice)
    
    with col2:
        st.header("📊 Model Status")
        
        # Model status indicators
        logreg_status = "✅ Loaded" if analyzer.logreg_model is not None else "❌ Not Loaded"
        distilbert_status = "✅ Loaded" if analyzer.distilbert_model is not None else "❌ Not Loaded"
        
        st.write(f"**Logistic Regression:** {logreg_status}")
        st.write(f"**DistilBERT:** {distilbert_status}")
        
        # Sample texts for testing
        st.subheader("🧪 Sample Texts")
        sample_texts = [
            "I am so happy today! This is wonderful!",
            "I feel really sad and disappointed.",
            "This makes me so angry and frustrated!",
            "I'm scared about what might happen.",
            "What a pleasant surprise! I didn't expect this.",
            "I love spending time with my family."
        ]
        
        selected_sample = st.selectbox("Choose a sample:", [""] + sample_texts)
        if selected_sample and st.button("📋 Use Sample"):
            st.session_state.sample_text = selected_sample
        
        # Dataset info
        st.subheader("📋 Dataset Requirements")
        st.markdown("""
        **CSV Format Required:**
        - Column 1: `text` (the text to analyze)
        - Column 2: `label` (emotion: joy, sadness, anger, fear, surprise, love)
        
        **Example:**
        ```
        text,label
        "I am happy",joy
        "I feel sad",sadness
        ```
        """)

def display_results(result, model_choice):
    """Display prediction results"""
    st.header("🎯 Prediction Results")
    
    # Main prediction
    st.success(f"**Predicted Emotion:** {result['prediction'].title()}")
    st.info(f"**Confidence:** {result['confidence']:.3f}")
    
    # Progress bar for confidence
    st.progress(result['confidence'])
    
    # Top predictions
    st.subheader("📈 All Emotion Probabilities")
    
    # Sort probabilities
    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
    
    # Display as columns
    cols = st.columns(min(3, len(sorted_probs)))
    for i, (emotion, prob) in enumerate(sorted_probs[:3]):
        with cols[i]:
            st.metric(
                label=emotion.title(),
                value=f"{prob:.3f}",
                delta=f"#{i+1}"
            )
    
    # Full probability chart
    if len(sorted_probs) > 3:
        prob_df = pd.DataFrame(sorted_probs, columns=['Emotion', 'Probability'])
        st.bar_chart(prob_df.set_index('Emotion'))
    
    # Show individual model results for hybrid
    if model_choice == "Hybrid (Combined)" and 'logreg_result' in result:
        st.subheader("🔍 Individual Model Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Logistic Regression:**")
            st.write(f"Prediction: {result['logreg_result']['prediction']}")
            st.write(f"Confidence: {result['logreg_result']['confidence']:.3f}")
        
        with col2:
            st.write("**DistilBERT:**")
            st.write(f"Prediction: {result['distilbert_result']['prediction']}")
            st.write(f"Confidence: {result['distilbert_result']['confidence']:.3f}")

if __name__ == "__main__":
    main()