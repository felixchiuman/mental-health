import gradio as gr
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

# Hugging Face login token (only needed for private models or DeepSeek)
import os
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Load PTSD summarization model
model_name = "machinelearningzuu/ptsd-summarization"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

# Sentiment analysis
sentiment_analyzer = pipeline("sentiment-analysis")

# Emotion classification
clf_model = AutoModelForSequenceClassification.from_pretrained("nateraw/bert-base-uncased-emotion")
clf_tokenizer = AutoTokenizer.from_pretrained("nateraw/bert-base-uncased-emotion")
labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

def classify_mental_state(text):
    inputs = clf_tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = clf_model(**inputs)
        logits = outputs.logits
        probs = torch.nn.functional.softmax(logits, dim=1)
        top_idx = torch.argmax(probs).item()
        label = labels[top_idx]
        confidence = probs[0][top_idx].item()
        return f"{label.capitalize()} ({confidence:.2f})"

# DeepSeek suggestion generator
deepseek_model_id = "deepseek-ai/deepseek-llm-7b-chat"
deepseek_tokenizer = AutoTokenizer.from_pretrained(deepseek_model_id)
deepseek_model = AutoModelForCausalLM.from_pretrained(
    deepseek_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    token=True
)
deepseek_tokenizer.pad_token = deepseek_tokenizer.eos_token

def generate_suggestion(summary_text):
    prompt = (
        f"Patient summary: {summary_text}\n"
        f"Based on this, provide 3 specific coping suggestions for PTSD symptoms:\n"
        f"1."
    )
    inputs = deepseek_tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(deepseek_model.device)
    outputs = deepseek_model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=200,
        do_sample=True,
        temperature=0.7,
        eos_token_id=deepseek_tokenizer.eos_token_id,
        pad_token_id=deepseek_tokenizer.pad_token_id
    )
    generated = deepseek_tokenizer.decode(outputs[0], skip_special_tokens=True)
    suggestion = generated.split("1.", 1)[-1].strip()
    return "1. " + suggestion

# Main logic for Gradio
def analyze_input(text):
    try:
        summary = summarizer(text, max_length=100, min_length=10, do_sample=False)[0]['summary_text']
        sentiment = sentiment_analyzer(text)[0]
        sentiment_result = f"{sentiment['label']} ({sentiment['score']:.2f})"
        classification_result = classify_mental_state(text)
        suggestion = generate_suggestion(summary)
        return summary, sentiment_result, classification_result, suggestion
    except Exception as e:
        return "Error: " + str(e), "Error", "Error", "Error"

# Gradio UI
demo = gr.Interface(
    fn=analyze_input,
    inputs=[gr.Textbox(lines=10, placeholder="Enter patient report...", label="Patient Report")],
    outputs=[
        gr.Textbox(label="Summary"),
        gr.Textbox(label="Sentiment Analysis"),
        gr.Textbox(label="Mental Health Indicator"),
        gr.Textbox(label="Suggested Advice")
    ],
    title="Mental Health Assistant",
    description="Summarizes PTSD-related text, detects emotional tone, classifies mental state, and generates non-clinical coping suggestions."
)

demo.launch()
