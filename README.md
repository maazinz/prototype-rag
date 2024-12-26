# README: RAG System Implementation

## Overview
This project implements a **Retrieval-Augmented Generation (RAG) System** using pre-trained models and external search tools. The system combines web search, semantic similarity ranking, and language generation to answer user queries effectively. It is designed to retrieve relevant documents, rank them by relevance, and generate contextual answers using a language model.

---

## Features

1. **Web Search Integration**:
   - Utilizes the DuckDuckGo search tool to fetch search results for a query.

2. **Document Ranking**:
   - Ranks retrieved documents using cosine similarity between query and document embeddings.

3. **Language Model Integration**:
   - Generates detailed answers by combining the query with the most relevant context using a pre-trained language model.

4. **Pipeline Execution**:
   - Complete workflow including search, retrieval, ranking, and answer generation in one function.

---

## Dependencies
The script requires the following libraries:

- `torch`
- `numpy`
- `transformers`
- `sentence_transformers`
- `langchain_community`

### Installing Required Libraries
Run the following command to install the dependencies:
```bash
pip install torch numpy transformers sentence_transformers langchain_community
```

---

## Code Components

### 1. **Initialization**
The `RAGSystem` class initializes with:
- **Embedding Model**: Default - `thenlper/gte-small`
- **Language Model**: Default - `unsloth/Llama-3.2-1B-Instruct`
- **Search Tool**: DuckDuckGoSearchResults from `langchain_community`.

### 2. **Web Search**
The `search_web` method retrieves up to `top_k` search results for a query.

### 3. **Ranking Documents**
The `rank_documents` method uses the cosine similarity between embeddings of the query and retrieved documents to rank them by relevance.

### 4. **Answer Generation**
The `generate_answer` method combines the query and the most relevant context to generate an answer using the language model.

### 5. **Full Pipeline Execution**
The `answer_question` method orchestrates the search, ranking, and answer generation steps into a single pipeline.

### 6. **Batch Testing**
The `run_rag` method runs the pipeline for a set of test questions or a custom query provided by the user.

---

## Usage
The file was runn in Google Colab to utilize the available GPU.
```python
from rag import RAGSystem
rag_test = RAGSystem()
rag_test.run_rag("What is Machine Learning?")
```

### 1. **Standalone Execution**
Run the script directly to execute the RAG pipeline for default test questions:
```bash
python rag.py
```

### 2. **Custom Query**
To test with a custom query, pass it to the `run_rag` function in the main block:
```python
runner.run_rag("What is the capital of France?")
```

---

## Example Output

### Input Query
```
What are the main causes of climate change?
```

### Output
#### Search Results:
1. Result 1: "Climate change is driven by..."
2. Result 2: "The primary causes include..."

#### Ranked Documents (Similarity Score):
- Score 0.98: "Greenhouse gas emissions are..."
- Score 0.85: "Burning fossil fuels..."

#### Generated Answer:
"The main causes of climate change include greenhouse gas emissions, deforestation, and industrial activities that increase carbon dioxide levels in the atmosphere."

---

## Future Improvements
- Allow integration with other search tools.
- Support multi-turn question answering.
- Optimize for large-scale datasets.

---
## PEP8 Guidelines
The python file has been checked and linted for PEP8 guidelines using flake8 and autopep8 to ensure that all coding standards are properly followed.

---

## Acknowledgements
- Pre-trained models from [Hugging Face](https://huggingface.co/).
- DuckDuckGo search integration via LangChain.