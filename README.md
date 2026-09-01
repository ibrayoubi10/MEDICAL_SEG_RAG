# MEDICAL_SEG_RAG

My first RAG pipeline project.

The goal is to collect arXiv papers related to medical image segmentation, U-Net variants, deep learning, computer vision, and medical imaging, then prepare them for a simple retrieval-augmented generation workflow.

The current script, `medical_segmentation_paper_collector.py`, queries the arXiv API, saves themed search results in `data/medical_segmentation/`, and maintains a deduplicated paper catalog for the next steps: PDF download, text extraction, chunking, embeddings, and vector search.

## Medical Segmentation Paper Corpus

Build a specialized corpus with at least 3500 arXiv papers:

```bash
.venv/bin/python medical_segmentation_paper_collector.py medical
```

By default, the collector requests 200 papers per theme at a time, then moves to
the next theme. If the corpus target is not reached after one pass, it continues
with the next arXiv result page for each theme.

The collector searches across medical image segmentation, U-Net variants, nnU-Net,
CNN/Transformer segmentation, TransUNet/UNETR/Swin-UNETR, medical data
augmentation, diffusion augmentation, semi-supervised and weakly-supervised
segmentation, self-supervised medical image analysis, domain adaptation,
federated learning, 3D medical segmentation, MRI/CT/ultrasound/X-ray
segmentation, Dice/IoU/Hausdorff metrics, uncertainty estimation, explainable AI,
and foundation models such as SAM, MedSAM, and Medical SAM 2.

The merged catalog is saved here:

```text
data/medical_segmentation/arxiv_papers_catalog.json
```

## First ChromaDB vector database

Build a persistent local Chroma collection from the merged arXiv catalog:

```bash
.venv/bin/python chroma_index.py build --reset
```

Search it from the command line:

```bash
.venv/bin/python chroma_index.py search "U-Net variants for medical image segmentation"
```

The database is stored in `data/medical_segmentation/chroma_db`. This first version
embeds paper titles, authors, categories, and abstracts with Chroma's default local
embedding function. Later, this can be swapped for a stronger embedding model.

## First RAG Answer Script

Ask a question over the Chroma corpus:

```bash
.venv/bin/python rag_answer.py "Which papers discuss U-Net for medical segmentation?"
```

By default, the script retrieves relevant papers and asks your local Ollama model
to answer with exact article-title citations:

```bash
.venv/bin/python rag_answer.py "Which papers discuss U-Net for medical segmentation?" --provider ollama
```

The default Ollama model is `llama3:latest`. You can change it with `--model`:

```bash
.venv/bin/python rag_answer.py "Which papers discuss U-Net?" --provider ollama --model llama3:latest
```

Without any LLM call, it can still print the retrieved sources, which is useful
for testing the retrieval step.

To test retrieval only:

```bash
.venv/bin/python rag_answer.py "data augmentation for medical image segmentation" --retrieval-only
```

## Graphic Interface

Install or update the project dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

Launch the local web interface:

```bash
.venv/bin/streamlit run app.py
```

Then open the local URL shown by Streamlit and ask your question directly in the
browser. The app uses Ollama locally and always generates an answer from the
retrieved articles.

# Author
Ibrahim Al Ayoubi
