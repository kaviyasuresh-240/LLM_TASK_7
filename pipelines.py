import os
import pandas as pd
from transformers import pipeline

# Optional: store Hugging Face cache inside your project folder
os.environ["HF_HOME"] = "D:/HUGGING_FACE_PIPELINES/cache"

# Load input text
df = pd.read_csv("dataset/input_text.csv")
text = df.loc[0, "text"]

print("Initializing pipelines...")
sentiment_pipeline = pipeline("sentiment-analysis")
summarization_pipeline = pipeline("text-generation", model="facebook/bart-large-cnn")
translation_pipeline = pipeline("text-generation", model="facebook/nllb-200-distilled-600M")
qa_pipeline = pipeline("text-generation", model="google/flan-t5-base")

# Sentiment Analysis (split text into chunks under 512 tokens)
print("Running Sentiment Analysis...")

# Split the article into sentences or smaller parts
chunks = text.split(". ")   # split by sentences
sentiment_results = []

for chunk in chunks:
    if chunk.strip():  # skip empty strings
        result = sentiment_pipeline(chunk[:500])  # truncate to safe length
        sentiment_results.append({"text": chunk, "label": result[0]['label'], "score": result[0]['score']})

pd.DataFrame(sentiment_results).to_csv("dataset/sentiment_results.csv", index=False)

# Summarization
print("Running Summarization...")
summary = summarization_pipeline(text, max_length=50, min_length=20, do_sample=False)
pd.DataFrame(summary).to_csv("dataset/summarized_text.csv", index=False)

# Translation (English → French)
print("Running Translation...")
translation = translation_pipeline(f"translate English to French: {text}", max_length=200)
pd.DataFrame(translation).to_csv("dataset/translated_text.csv", index=False)

# Question Answering (single example)
print("Running Question Answering...")
qa_prompt = f"Question: What are some key applications of AI in medical imaging? Context: {text} Answer:"
qa_result = qa_pipeline(qa_prompt, max_length=64)
pd.DataFrame(qa_result).to_csv("dataset/qa_results.csv", index=False)

# Comparison table
print("Creating comparison table...")

# Compute overall sentiment trend (majority label)
labels = [r["label"] for r in sentiment_results]
overall_sentiment = max(set(labels), key=labels.count)  # most frequent label

comparison = {
    "Original Text": [text],
    "Summary": [summary[0]['generated_text']],
    "Translation": [translation[0]['generated_text']],
    "Sentiment": [overall_sentiment],
    "QA Answer": [qa_result[0]['generated_text']]
}

pd.DataFrame(comparison).to_csv("dataset/pipeline_comparison.csv", index=False)

# -------------------------------------------------------------------
# Experiments 
# -------------------------------------------------------------------

# 1. Compare Summaries with Different max_length Values
print("Running Extended Summarization Comparison...")
summary_short = summarization_pipeline(text, max_length=30, min_length=10, do_sample=False)
summary_long = summarization_pipeline(text, max_length=80, min_length=30, do_sample=False)

summary_compare = {
    "Short Summary": [summary_short[0]['generated_text']],
    "Long Summary": [summary_long[0]['generated_text']]
}
pd.DataFrame(summary_compare).to_csv("dataset/summarization_comparison.csv", index=False)

# 2. Translate into Two Languages (French + Spanish)
print("Running Extended Translation...")
translation_fr = translation_pipeline(f"translate English to French: {text}", max_length=200)
translation_es = translation_pipeline(f"translate English to Spanish: {text}", max_length=200)

translation_compare = {
    "French Translation": [translation_fr[0]['generated_text']],
    "Spanish Translation": [translation_es[0]['generated_text']]
}
pd.DataFrame(translation_compare).to_csv("dataset/translation_comparison.csv", index=False)

# 3. Create Five QA Prompts (tailored to AI in Healthcare)
print("Running Extended Question Answering...")
questions = [
    "What are some key applications of AI in medical imaging?",
    "How does AI help in predicting patient health risks?",
    "In what ways is AI improving hospital operations and administration?",
    "What ethical challenges are associated with using AI in healthcare?",
    "How does AI empower doctors rather than replace them?"
]

qa_answers = []
for q in questions:
    prompt = f"Question: {q} Context: {text} Answer:"
    result = qa_pipeline(prompt, max_length=64)
    qa_answers.append({"Question": q, "Answer": result[0]['generated_text']})

qa_df = pd.DataFrame(qa_answers)
qa_df.to_csv("dataset/qa_multiple_results.csv", index=False)

print("Experiments completed and saved in dataset/")







