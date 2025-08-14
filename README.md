# sentiment-analysis-hybrid-model
Hybrid Sentiment Analysis System combining Logistic Regression and DistilBERT for efficient and accurate classification of social media feedback, built as part of my TIPE project after CPGE.

A **comprehensive sentiment analysis** tool comparing three approaches: **Logistic Regression**, **DistilBERT**, and a **Hybrid model** combining both.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)  
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)  
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)  
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)  

---
<img width="1783" height="931" alt="image" src="https://github.com/user-attachments/assets/72625cc4-4d7f-43ba-b895-8fbd8a5678ac" />


## 🌟 Live Demo  
**Coming soon!**  
After deployment



---

## ✨ Features  

- **Three Models:**  
  - 🤖 Logistic Regression (TF-IDF)  
  - 🧠 DistilBERT Transformer  
  - 🔄 Hybrid (weighted combination)  

- **Interactive UI:**  
  - Real-time sentiment & emotion detection  
  - Confidence scores + probability charts  
  - Adjustable model weights  
  - Built-in sample texts  

- **Emotions Detected:**  
  Joy • Sadness • Anger • Fear • Surprise • Love  

---

## 📊 Dataset  

This project uses the **[Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp)** by [Praveen Govi](https://www.kaggle.com/praveengovi),  
licensed under **[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**.  

- **Changes made:** cleaned text, split into train/test sets, and adapted format for model training.  
- Under the CC BY-SA license:  
  - ✅ You are free to use, share, and adapt the dataset, even commercially.  
  - ⚠️ You must give attribution, include a link to the license, and indicate modifications.  
  - 🔄 If you redistribute it, you must use the same license.  

---

## 🚀 Quick Start  

### 🔹 Run Online  
Go to **[Sentiment Analysis App](https://your-username-sentiment-analysis.streamlit.app)**  

### 🔹 Run Locally  
```bash
# 1️⃣ Clone repo
git clone https://github.com/your-username/sentiment-analysis-app.git
cd sentiment-analysis-app

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Launch app
streamlit run app.py
````

Then open **[http://localhost:8501](http://localhost:8501)** in your browser.

---

## 📊 Models Overview

| Model               | Pros                         | Cons                | Best For                 |
| ------------------- | ---------------------------- | ------------------- | ------------------------ |
| Logistic Regression | Fast, lightweight            | Needs training data | Quick deployments        |
| DistilBERT          | High accuracy, context-aware | Slower, large model | Precision-critical tasks |
| Hybrid              | Balanced performance         | More complex        | Mixed needs              |

---

## 📁 Structure

```
sentiment-analysis-app/
├── app.py               # Main Streamlit app
├── requirements.txt     # Dependencies
├── models/              # Saved models
├── data/                # Sample data
└── README.md
```

---

## 🛠️ Technical Stack

* **Frontend:** Streamlit
* **ML:** scikit-learn, transformers, torch
* **Models:**

  * Logistic Regression + TF-IDF
  * DistilBERT (`bhadresh-savani/distilbert-base-uncased-emotion`)
* **Deployment:** Streamlit Cloud


> *Results may vary with dataset & text domain.*

---

## 🤝 Contributing

1. Fork the repo
2. Create a branch (`git checkout -b feature/new-feature`)
3. Commit (`git commit -m "Add new feature"`)
4. Push (`git push origin feature/new-feature`)
5. Open a PR 🚀

---

## 📝 License

This project is licensed under the MIT License – see [LICENSE](LICENSE).
The dataset used is licensed separately under [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) by Praveen Govi.

---

## 🙏 Credits

* Dataset: [Emotions Dataset for NLP](https://www.kaggle.com/datasets/praveengovi/emotions-dataset-for-nlp) by Praveen Govi (CC BY-SA 4.0)
* [DistilBERT](https://huggingface.co/bhadresh-savani/distilbert-base-uncased-emotion) – bhadresh-savani
* [Streamlit](https://streamlit.io/)
* [Hugging Face Transformers](https://huggingface.co/transformers/)

---

## 📧 Contact
Zineb
zinebelrhiti111@gmail.com
https://github.com/zinebev

---

⭐ **If you like this project, give it a star on GitHub!** ⭐

