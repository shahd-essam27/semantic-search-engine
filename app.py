"""Milestone 6 Gradio semantic search app. Run: python app.py"""
from __future__ import annotations
from pathlib import Path
from src.data_loader import DEFAULT_DATASET, load_documents
from src.search import search_documents


def categories() -> list[str]:
    try:
        return sorted({row['category'] for row in load_documents(DEFAULT_DATASET)})
    except (FileNotFoundError, ValueError):
        return []


def ui_search(query: str, top_k: int, category: str, date_from: str, date_to: str):
    """Return a status message and a table without raising UI exceptions."""
    if not query or not query.strip():
        return 'Please type a search query.', []
    try:
        results = search_documents(query, int(top_k), category or None,
                                   date_from.strip() or None, date_to.strip() or None)
    except ValueError as exc:
        return f'Invalid input: {exc}', []
    except Exception as exc:
        # Do not expose internal exception details to public UI.
        return ('Search is unavailable. Make sure you have indexed the dataset '
                'with: python -m scripts.build_index (and check terminal logs).'), []
    if not results:
        return 'No matching results. Check the filters, or build the index first.', []
    table = [[i, hit['document_id'], hit['similarity'], hit['category'],
              hit['date'], hit['source'], hit['text']] for i, hit in enumerate(results, 1)]
    return f'Found {len(results)} matching chunks, ranked by cosine similarity.', table


def build_ui():
    import gradio as gr
    with gr.Blocks(title='Week 5 Semantic Search Engine') as demo:
        gr.Markdown('# Semantic Search Engine\nMeaning-based search using local embeddings and Chroma. This retrieves text; it does not generate answers.')
        with gr.Row():
            query = gr.Textbox(label='Search query', placeholder='e.g., How can I protect my online accounts?', scale=3)
            top_k = gr.Slider(1, 20, step=1, value=5, label='Top-k results', scale=1)
        with gr.Row():
            category = gr.Dropdown(choices=[''] + categories(), value='', label='Category (optional)')
            date_from = gr.Textbox(label='From date (YYYY-MM-DD)', placeholder='2026-01-01')
            date_to = gr.Textbox(label='To date (YYYY-MM-DD)', placeholder='2026-12-31')
        submit = gr.Button('Search', variant='primary')
        status = gr.Markdown()
        results = gr.Dataframe(headers=['Rank', 'Document ID', 'Similarity', 'Category', 'Date', 'Source', 'Text'],
                               datatype=['number', 'str', 'number', 'str', 'str', 'str', 'str'],
                               interactive=False, wrap=True, label='Ranked results')
        submit.click(ui_search, inputs=[query, top_k, category, date_from, date_to], outputs=[status, results])
        query.submit(ui_search, inputs=[query, top_k, category, date_from, date_to], outputs=[status, results])
        gr.Markdown('**Setup:** Run `python -m scripts.build_index` before launching the app. Models and indexed vectors stay local.')
    return demo


if __name__ == '__main__':
    build_ui().launch(inbrowser=True)
