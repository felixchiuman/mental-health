# 🧠 Mental Health Assistant: PTSD-Aware Text Analysis

**A mental health NLP tool that summarizes trauma narratives, detects emotional tone, classifies psychological state, and provides LLM-generated coping suggestions for PTSD.**

---

## 📌 Description

This project uses state-of-the-art NLP models to assist with mental health support by analyzing clinical or trauma-related patient reports.  
It aims to provide summarized insights, emotional profiling, and self-care strategies based on the content, using Hugging Face Transformers and Gradio.

---

## 📌 API Code
You can access the API Code project here: [Mental Health API](https://github.com/felixchiuman/mental-health-api)

---

## 💡 Features

| Task                      | Model Used                                                                 |
|---------------------------|----------------------------------------------------------------------------|
| 📝 Summarization             | `machinelearningzuu/ptsd-summarization`                                   |
| 📊 Sentiment Analysis        | Hugging Face `sentiment-analysis` pipeline                                |
| 💡 Emotion Classification    | `nateraw/bert-base-uncased-emotion`                                       |
| 🛠 PTSD Coping Suggestions  | `deepseek-ai/deepseek-llm-7b-chat`                                         |

---

## 🚀 Live Demo

👉 **Launch on Hugging Face Spaces:**  
[https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME](https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME)

---

## 🔐 Security Note

This app uses a Hugging Face token to access gated models like DeepSeek.

Use Hugging Face **Secrets** for safe token management (in Spaces dashboard):

```python
import os
from huggingface_hub import login
login(os.getenv("HF_TOKEN"))
