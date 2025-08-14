# sentiment_analysis_app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import torch
import os
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

# Configuration
MODEL_PATHS = {
    'logreg_model': './models/logreg_model.joblib',
    'tfidf_vectorizer': './models/tfidf_vectorizer.joblib',
    'distilbert_model': 'bhadresh-savani/distilbert-base-uncased-emotion'
}

# Emotion labels mapping for consistency
EMOTION_LABELS = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

class SentimentAnalyzer:
    """
    Main class for sentiment analysis with three different models:
    1. Logistic Regression with TF-IDF
    2. DistilBERT pretrained
    3. Hybrid model combining both
    """
    
    def __init__(self):
        self.logreg_model = None
        self.tfidf_vectorizer = None
        self.distilbert_model = None
        self.distilbert_tokenizer = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def load_or_train_logistic_regression(self, train_data_path=None):
        """
        Load pre-trained Logistic Regression model or train a new one
        """
        os.makedirs('./models', exist_ok=True)
        
        if os.path.exists(MODEL_PATHS['logreg_model']) and os.path.exists(MODEL_PATHS['tfidf_vectorizer']):
            st.info("Loading saved Logistic Regression model...")
            self.logreg_model = joblib.load(MODEL_PATHS['logreg_model'])
            self.tfidf_vectorizer = joblib.load(MODEL_PATHS['tfidf_vectorizer'])
            return True
        
        if train_data_path and os.path.exists(train_data_path):
            st.info("Training new Logistic Regression model...")
            return self._train_logistic_regression(train_data_path)
        else:
            st.warning("No pre-trained Logistic Regression model found and no training data provided.")
            return False
    
    def _train_logistic_regression(self, data_path):
        """
        Train Logistic Regression on the provided dataset
        """
        try:
            # Load data
            texts, labels = [], []
            with open(data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split(';')
                    if len(parts) == 2:
                        texts.append(parts[0])
                        labels.append(parts[1])
            
            if not texts:
                st.error("No valid data found in the file.")
                return False
            
            # Train/test split
            X_train, X_test, y_train, y_test = train_test_split(
                texts, labels, test_size=0.2, random_state=42
            )
            
            # TF-IDF Vectorization
            self.tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
            
            # Train Logistic Regression
            self.logreg_model = LogisticRegression(max_iter=1000, random_state=42)
            self.logreg_model.fit(X_train_tfidf, y_train)
            
            # Evaluate
            X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
            y_pred = self.logreg_model.predict(X_test_tfidf)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Save models
            joblib.dump(self.logreg_model, MODEL_PATHS['logreg_model'])
            joblib.dump(self.tfidf_vectorizer, MODEL_PATHS['tfidf_vectorizer'])
            
            st.success(f"Logistic Regression trained successfully! Accuracy: {accuracy:.4f}")
            return True
            
        except Exception as e:
            st.error(f"Error training Logistic Regression: {str(e)}")
            return False
    
    def load_distilbert(self):
        """
        Load pre-trained DistilBERT model for emotion classification
        """
        try:
            st.info("Loading DistilBERT model...")
            self.distilbert_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATHS['distilbert_model'])
            self.distilbert_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATHS['distilbert_model'])
            self.distilbert_model.to(self.device)
            st.success("DistilBERT model loaded successfully!")
            return True
        except Exception as e:
            st.error(f"Error loading DistilBERT: {str(e)}")
            return False
    
    def predict_logistic_regression(self, text):
        """
        Predict sentiment using Logistic Regression + TF-IDF
        """
        if self.logreg_model is None or self.tfidf_vectorizer is None:
            return None, None
        
        try:
            # Vectorize input text
            text_tfidf = self.tfidf_vectorizer.transform([text])
            
            # Get prediction and probabilities
            prediction = self.logreg_model.predict(text_tfidf)[0]
            probabilities = self.logreg_model.predict_proba(text_tfidf)[0]
            
            # Get confidence score
            confidence = np.max(probabilities)
            
            # Create probability distribution
            prob_dict = {
                label: prob for label, prob in zip(self.logreg_model.classes_, probabilities)
            }
            
            return prediction, confidence, prob_dict
            
        except Exception as e:
            st.error(f"Error in Logistic Regression prediction: {str(e)}")
            return None, None, None
    
    def predict_distilbert(self, text):
        """
        Predict sentiment using DistilBERT
        """
        if self.distilbert_model is None or self.distilbert_tokenizer is None:
            return None, None
        
        try:
            # Tokenize input
            inputs = self.distilbert_tokenizer(text, return_tensors="pt", truncation=True, 
                                             padding=True, max_length=512)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Get prediction
            with torch.no_grad():
                outputs = self.distilbert_model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class_id = probabilities.argmax().item()
                confidence = probabilities.max().item()
            
            # Map to emotion label
            emotion_labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
            prediction = emotion_labels[predicted_class_id]
            
            # Create probability distribution
            prob_dict = {
                label: prob.item() for label, prob in zip(emotion_labels, probabilities[0])
            }
            
            return prediction, confidence, prob_dict
            
        except Exception as e:
            st.error(f"Error in DistilBERT prediction: {str(e)}")
            return None, None, None
    
    def predict_hybrid(self, text, lr_weight=0.5, bert_weight=0.5):
        """
        Predict sentiment using hybrid approach (weighted combination)
        """
        # Get predictions from both models
        lr_pred, lr_conf, lr_probs = self.predict_logistic_regression(text)
        bert_pred, bert_conf, bert_probs = self.predict_distilbert(text)
        
        if lr_pred is None or bert_pred is None:
            return None, None, None
        
        try:
            # Get common emotion labels
            common_labels = set(lr_probs.keys()) & set(bert_probs.keys())
            
            if not common_labels:
                # If no common labels, return the more confident prediction
                if lr_conf > bert_conf:
                    return lr_pred, lr_conf, lr_probs
                else:
                    return bert_pred, bert_conf, bert_probs
            
            # Weighted combination of probabilities
            hybrid_probs = {}
            for label in common_labels:
                lr_prob = lr_probs.get(label, 0)
                bert_prob = bert_probs.get(label, 0)
                hybrid_probs[label] = (lr_weight * lr_prob) + (bert_weight * bert_prob)
            
            # Get final prediction
            final_prediction = max(hybrid_probs, key=hybrid_probs.get)
            final_confidence = hybrid_probs[final_prediction]
            
            return final_prediction, final_confidence, hybrid_probs
            
        except Exception as e:
            st.error(f"Error in hybrid prediction: {str(e)}")
            return None, None, None

def main():
    """
    Main Streamlit application
    """
    st.set_page_config(
        page_title="Sentiment Analysis App",
        page_icon="😊",
        layout="wide"
    )
    
    st.title("🎭 Advanced Sentiment Analysis App")
    st.markdown("""
    This application compares three different approaches for sentiment/emotion analysis:
    - **Logistic Regression** with TF-IDF features
    - **DistilBERT** pretrained model
    - **Hybrid Model** combining both approaches
    """)
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = SentimentAnalyzer()
    
    analyzer = st.session_state.analyzer
    
    # Sidebar for model setup
    st.sidebar.header("🔧 Model Setup")
    
    # Setup Logistic Regression
    st.sidebar.subheader("Logistic Regression Setup")
    
    # Option to upload training data
    uploaded_file = st.sidebar.file_uploader(
        "Upload training data (optional)", 
        type=['txt'],
        help="Upload a text file with format: text;label per line"
    )
    
    if uploaded_file is not None:
        # Save uploaded file temporarily
        temp_path = "./temp_train_data.txt"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        if st.sidebar.button("Train Logistic Regression"):
            analyzer.load_or_train_logistic_regression(temp_path)
    else:
        if st.sidebar.button("Load Pre-trained Logistic Regression"):
            analyzer.load_or_train_logistic_regression()
    
    # Setup DistilBERT
    st.sidebar.subheader("DistilBERT Setup")
    if st.sidebar.button("Load DistilBERT Model"):
        analyzer.load_distilbert()
    
    # Main interface
    st.header("📝 Text Analysis")
    
    # Text input
    input_text = st.text_area(
        "Enter text to analyze:",
        height=100,
        placeholder="Type your text here... (e.g., 'I am so happy today!')"
    )
    
    # Model selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        selected_model = st.selectbox(
            "Select Model:",
            ["Logistic Regression", "DistilBERT", "Hybrid Model"]
        )
    
    with col2:
        if selected_model == "Hybrid Model":
            lr_weight = st.slider("Logistic Regression Weight", 0.0, 1.0, 0.5, 0.1)
            bert_weight = 1.0 - lr_weight
            st.write(f"DistilBERT Weight: {bert_weight:.1f}")
    
    # Prediction button
    if st.button("🔍 Analyze Sentiment", type="primary"):
        if not input_text.strip():
            st.warning("Please enter some text to analyze.")
        else:
            with st.spinner("Analyzing..."):
                start_time = time.time()
                
                # Get prediction based on selected model
                if selected_model == "Logistic Regression":
                    if analyzer.logreg_model is None:
                        st.error("Logistic Regression model not loaded. Please load or train the model first.")
                    else:
                        pred, conf, probs = analyzer.predict_logistic_regression(input_text)
                
                elif selected_model == "DistilBERT":
                    if analyzer.distilbert_model is None:
                        st.error("DistilBERT model not loaded. Please load the model first.")
                    else:
                        pred, conf, probs = analyzer.predict_distilbert(input_text)
                
                else:  # Hybrid Model
                    if analyzer.logreg_model is None or analyzer.distilbert_model is None:
                        st.error("Both models must be loaded for hybrid prediction.")
                    else:
                        pred, conf, probs = analyzer.predict_hybrid(input_text, lr_weight, bert_weight)
                
                end_time = time.time()
                
                # Display results
                if pred is not None:
                    st.success("Analysis Complete!")
                    
                    # Main result
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Predicted Emotion", pred.capitalize())
                    with col2:
                        st.metric("Confidence Score", f"{conf:.3f}")
                    with col3:
                        st.metric("Processing Time", f"{(end_time - start_time):.3f}s")
                    
                    # Detailed probabilities
                    if probs:
                        st.subheader("📊 Detailed Probability Scores")
                        
                        # Create DataFrame for better display
                        prob_df = pd.DataFrame([
                            {"Emotion": emotion.capitalize(), "Probability": prob}
                            for emotion, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True)
                        ])
                        
                        # Display as bar chart
                        st.bar_chart(prob_df.set_index('Emotion')['Probability'])
                        
                        # Display as table
                        st.dataframe(prob_df, use_container_width=True)
    
    # Example texts section
    st.header("💡 Try These Examples")
    examples = [
        ("I am absolutely thrilled about this amazing opportunity!", "joy"),
        ("This is the worst day of my life, everything is going wrong.", "sadness"),
        ("I can't believe you would do such a thing to me!", "anger"),
        ("I'm so scared about what might happen tomorrow.", "fear"),
        ("Wow, I never expected this to happen!", "surprise"),
        ("I love spending time with my family on weekends.", "love")
    ]
    
    cols = st.columns(2)
    for i, (example_text, expected) in enumerate(examples):
        with cols[i % 2]:
            if st.button(f"Example {i+1}: {expected.capitalize()}", key=f"example_{i}"):
                st.text_area("Example text:", value=example_text, key=f"example_text_{i}")
    
    # Model information
    with st.expander("ℹ️ Model Information"):
        st.markdown("""
        ### Model Details
        
        **Logistic Regression + TF-IDF:**
        - Uses TF-IDF (Term Frequency-Inverse Document Frequency) features
        - Fast and interpretable
        - Good baseline performance
        - Requires training on your specific dataset
        
        **DistilBERT:**
        - Pre-trained transformer model (distilled version of BERT)
        - State-of-the-art performance
        - Understands context and semantics
        - Ready to use without additional training
        
        **Hybrid Model:**
        - Combines predictions from both models
        - Uses weighted averaging of probability scores
        - Can leverage strengths of both approaches
        - Customizable weight distribution
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with Streamlit • Powered by scikit-learn and Transformers")

if __name__ == "__main__":
    main()