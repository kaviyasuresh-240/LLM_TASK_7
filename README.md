```markdown
# AI in Healthcare – Hugging Face NLP Pipelines

An end-to-end NLP workflow applying Hugging Face pipelines to analyze medical and healthcare text through sentiment analysis, summarization, translation, and question answering.

## 🧠 Models Used

* **Sentiment Analysis:** `distilbert-base-uncased-finetuned-sst-2-english`
* **Summarization:** `facebook/bart-large-cnn`
* **Translation:** `facebook/nllb-200-distilled-600M`
* **Question Answering:** `google/flan-t5-base`

## 📂 Project Structure & Outputs

All generated CSV outputs are stored in `dataset/`:

* **Core Tasks:** `sentiment_results.csv`, `summarized_text.csv`, `translated_text.csv`, `qa_results.csv`, `pipeline_comparison.csv`
* **Extended Experiments:** `summarization_comparison.csv`, `translation_comparison.csv`, `qa_multiple_results.csv`

## 🚀 Quickstart

```bash
git clone https://github.com/kaviyasuresh-240/LLM_TASK_7.git
cd LLM_TASK_7
pip install -r requirements.txt
python pipelines.py
```
```
Note: Open Task 7_HuggingFace_Pipelines.ipynb for detailed walkthroughs, visualizations, and extended experiment analysis.

## 🧠 Future Work

- **Evaluation Metrics:** Integrate ROUGE (summarization), BLEU (translation), and F1/Exact Match (QA) to evaluate model outputs quantitatively.  
- **Model Scaling:** Experiment with larger foundation models (e.g., T5-Large, GPT-Neo, BioBERT) to improve clinical precision.  
- **Interactive UI:** Deploy the end-to-end pipeline as a real-time web application using Streamlit or Gradio.  
- **Domain Adaptation:** Fine-tune models on biomedical datasets (like PubMed or MIMIC-III) for specialized healthcare applications.
