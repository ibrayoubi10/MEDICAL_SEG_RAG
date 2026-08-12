# Arxiv_RAG

My first RAG pipeline project.

The goal is to collect arXiv papers related to AI, machine learning, NLP, and information retrieval, then prepare them for a simple retrieval-augmented generation workflow.

The current script, `arxiv_paper_collector.py`, queries the arXiv API, saves search results in `data/`, and maintains a deduplicated paper catalog for the next steps: PDF download, text extraction, chunking, embeddings, and vector search.
