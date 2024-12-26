import torch
import numpy as np
from transformers import pipeline
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
from langchain_community.tools import DuckDuckGoSearchResults

class RAGSystem:
    def __init__(self, 
                 embedding_model='thenlper/gte-small', 
                 llm_model='unsloth/Llama-3.2-1B-Instruct'):
        """
        Initialize RAG system with embedding and language models
        
        Args:
            embedding_model (str): Path to embedding model
            llm_model (str): Path to language model
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize embedding model
        self.embedding_model = SentenceTransformer(embedding_model, device=str(device))
        
        # Initialize LLM
        self.llm_pipe = pipeline(
            "text-generation",
            model=llm_model,
            torch_dtype=torch.bfloat16,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        
        # Initialize search tool
        self.search = DuckDuckGoSearchResults(output_format="list")
    
    def search_web(self, query, top_k=4):
        """
        Search web for relevant documents
        
        Args:
            query (str): Search query
            top_k (int): Number of top results to retrieve
        
        Returns:
            list: Top search results
        """
        try:
            results = self.search.invoke(query)
            return results[:top_k]
        except Exception as e:
            print(f"Search error: {e}")
            return []
    
    def rank_documents(self, query, documents):
        """
        Rank documents by semantic similarity to query
        
        Args:
            query (str): Search query
            documents (list): List of document texts
        
        Returns:
            list: Ranked documents with similarity scores
        """
        # Embed query and documents
        query_embedding = self.embedding_model.encode(query)
        doc_embeddings = self.embedding_model.encode(documents)
        
        # Calculate cosine similarities
        similarities = [cos_sim(query_embedding, doc_emb)[0][0] for doc_emb in doc_embeddings]
        
        # Rank documents by similarity
        ranked_docs = sorted(
            zip(documents, similarities), 
            key=lambda x: x[1], 
            reverse=True
        )
        
        return ranked_docs
    
    def generate_answer(self, query, context):
        """
        Generate answer using LLM with retrieved context
        
        Args:
            query (str): Original query
            context (str): Context retrieved from search
        
        Returns:
            str: Generated answer
        """
        messages = [
            {
                "role": "system", 
                "content": "You are a helpful assistant that provides accurate answers based on provided context."
            },
            {
                "role": "user", 
                "content": f"Question: {query}\n\nContext: {context}"
            }
        ]
        
        outputs = self.llm_pipe(
            messages,
            max_new_tokens=512,
        )
        
        return outputs[0]["generated_text"][-1]["content"]
    
    def answer_question(self, query):
        """
        Full RAG pipeline: search, retrieve, rank, and generate answer
        
        Args:
            query (str): Question to answer
        
        Returns:
            dict: Comprehensive RAG result
        """
        # Search web
        search_results = self.search_web(query)
        
        # Extract texts from search results
        result_texts = [result['snippet'] for result in search_results]
        
        # Rank documents
        ranked_docs = self.rank_documents(query, result_texts)
        
        # Choose top document as context
        top_context = ranked_docs[0][0] if ranked_docs else ""
        
        # Generate answer
        answer = self.generate_answer(query, top_context)
        
        return {
            "query": query,
            "search_results": search_results,
            "ranked_documents": ranked_docs,
            "answer": answer
        }

    def run_rag(self, input_question=None):       
        if input_question:
            test_questions = []
            test_questions.append(input_question)
        else:
            test_questions = [
                "Why did China enact the one child policy?",
                "What are the main causes of climate change?", 
                "How does machine learning work?",
                "What was the impact of the Industrial Revolution?",
                "Explain the basic principles of quantum computing"
            ]
        
        for question in test_questions:
            result = self.answer_question(question)
            print("\n--- Question:", question)
            print("Search Results:")
            for i, doc in enumerate(result['search_results'], 1):
                print(f"{i}. {doc['title']}: {doc['snippet']}")
            
            print("\nRanked Documents (Similarity Score):")
            for doc, score in result['ranked_documents']:
                print(f"- Score {score:.4f}: {doc[:200]}...")
            
            print("\nGenerated Answer:", result['answer'])
            print("-" * 50)

if __name__ == "__main__":
    runner = RAGSystem()
    runner.run_rag() # You can also add your custom question here as in the notebook