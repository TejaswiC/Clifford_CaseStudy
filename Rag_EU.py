# =========================================================
# IMPORTS
# =========================================================

import re
import numpy as np

from transformers import pipeline
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline

from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun

from pydantic import ConfigDict
from typing import List


# =========================================================
# 1. CLEAN TEXT FUNCTION
# =========================================================

def clean_text(text):

    # Remove hyphenated line breaks
    text = re.sub(r'-\s*\n\s*', '', text)

    # Replace line breaks with spaces
    text = text.replace('\n', ' ')

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    return text.strip()


# =========================================================
# 2. LOAD PDF
# =========================================================

print("\nLoading PDF...\n")

loader = PyPDFLoader("EU_GDPR.pdf")

documents = loader.load()

print(f"Loaded {len(documents)} pages")


# =========================================================
# 3. CLEAN DOCUMENTS
# =========================================================

print("\nCleaning extracted text...\n")

for doc in documents:
    doc.page_content = clean_text(doc.page_content)


# =========================================================
# 4. CHUNK SIZE BENCHMARKING
# =========================================================

CHUNK_CONFIGS = [
    {"chunk_size": 512,  "chunk_overlap": 128},
    {"chunk_size": 800,  "chunk_overlap": 200},
    {"chunk_size": 1200, "chunk_overlap": 300}
]

all_chunk_sets = {}

for config in CHUNK_CONFIGS:

    print(
        f"\nCreating chunks: "
        f"size={config['chunk_size']} / overlap={config['chunk_overlap']}"
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["chunk_size"],
        chunk_overlap=config["chunk_overlap"]
    )

    chunked_docs = splitter.split_documents(documents)

    for idx, doc in enumerate(chunked_docs):
        doc.metadata["chunk_id"]   = idx
        doc.metadata["chunk_size"] = config["chunk_size"]
        doc.metadata["page_number"] = doc.metadata.get("page", "Unknown")

    key = f"{config['chunk_size']}_{config['chunk_overlap']}"
    all_chunk_sets[key] = chunked_docs

    print(f"  Chunks created: {len(chunked_docs)}")

# =========================================================
# SELECT PRODUCTION CONFIG
# =========================================================

# 800/200 balances context richness vs token limit safety
docs = all_chunk_sets["800_200"]

print(f"\nSelected production config: 800 / 200")
print(f"Total chunks in use: {len(docs)}")


# =========================================================
# 5. LOAD EMBEDDING MODEL
# =========================================================

print("\nLoading embedding model...\n")

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)


# =========================================================
# 6. CREATE FAISS VECTOR STORE
# =========================================================

print("\nCreating FAISS vector store...\n")

vectorstore = FAISS.from_documents(docs, embeddings)

print(f"Vector store created with {len(docs)} documents")


# =========================================================
# 7. CREATE BM25 INDEX
# =========================================================

print("\nCreating BM25 index...\n")

bm25_corpus = [doc.page_content for doc in docs]

tokenized_corpus = [
    re.findall(r"\w+", doc.lower())
    for doc in bm25_corpus
]

bm25 = BM25Okapi(tokenized_corpus)

print(f"BM25 index created with {len(tokenized_corpus)} documents")


# =========================================================
# 8. LOAD RERANKER
# =========================================================

print("\nLoading reranker...\n")

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

print("Reranker loaded")


# =========================================================
# 9. LOCAL LLM
# =========================================================

print("\nLoading local LLM (flan-t5-base)...\n")

hf_pipeline = pipeline(
    task="text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200,
    truncation=True,
    do_sample=False
)

llm = HuggingFacePipeline(pipeline=hf_pipeline)

print("LLM loaded")


# =========================================================
# 10. CATEGORY-AWARE PROMPT TEMPLATES
# =========================================================
PROMPT_TEMPLATES = {

    "fact": """Answer ONLY from the provided legal context.
Give a concise factual answer, and cite article numbers explicitly if available.

Context:
{context}

Question:
{query}

Answer:""",

    "abstract": """Explain the concept clearly and in depth,
based ONLY on the principles found in the provided context.
Avoid references to external laws unless explicitly mentioned.

Context:
{context}

Question:
{query}

Answer:""",

    "reasoning": """Analyze step-by-step which articles or provisions apply.
Explain the reasoning clearly, and cite article numbers where possible.
If multiple rules are triggered, list them in order.

Context:
{context}

Question:
{query}

Answer:""",

    "comparative": """Compare the frameworks point by point,
using ONLY the provided context. 
If the context covers only one framework, state that explicitly instead of inventing details.

Context:
{context}

Question:
{query}

Answer:"""
}

# =========================================================
# 11. TOKEN-SAFE CONTEXT BUILDER
# =========================================================

def build_context(retrieved_docs, max_chars=2000):

    context_parts = []
    current_length = 0

    for doc in retrieved_docs:

        chunk = doc.page_content

        if current_length + len(chunk) > max_chars:
            break

        context_parts.append(chunk)
        current_length += len(chunk)

    return "\n\n".join(context_parts)


# =========================================================
# 12. LOST-IN-THE-MIDDLE MITIGATION
# =========================================================


def reorder_for_long_context(docs):

    if len(docs) <= 2:
        return docs

    reordered = []
    left  = 0
    right = len(docs) - 1

    while left <= right:
        reordered.append(docs[left])
        if left != right:
            reordered.append(docs[right])
        left  += 1
        right -= 1

    return reordered


# =========================================================
# 13. RETRIEVAL MODES
# =========================================================

def dense_search(query, top_k=8):

    results = vectorstore.similarity_search_with_score(query, k=top_k)
    return [doc for doc, score in results]


def sparse_search(query, top_k=8):

    tokenized_query = re.findall(r"\w+", query.lower())
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [docs[i] for i in top_idx]


def hybrid_search(query, top_k=8):

    # --- Dense ---
    dense_results = vectorstore.similarity_search_with_score(query, k=top_k)

    dense_docs   = []
    dense_scores = []

    for doc, score in dense_results:
        dense_docs.append(doc)
        dense_scores.append(1 / (1 + score))

    dense_scores = np.array(dense_scores)
    dense_scores = (dense_scores - dense_scores.min()) / \
                   (dense_scores.max() - dense_scores.min() + 1e-8)

    # --- Sparse ---
    tokenized_query = re.findall(r"\w+", query.lower())
    bm25_scores     = bm25.get_scores(tokenized_query)
    top_bm25_idx    = np.argsort(bm25_scores)[::-1][:top_k]

    sparse_docs   = [docs[i] for i in top_bm25_idx]
    sparse_scores = np.array([bm25_scores[i] for i in top_bm25_idx])
    sparse_scores = (sparse_scores - sparse_scores.min()) / \
                    (sparse_scores.max() - sparse_scores.min() + 1e-8)

    # --- Weighted Fusion (50/50) ---
    combined = {}

    for doc, score in zip(dense_docs, dense_scores):
        key = doc.page_content
        combined[key] = {"doc": doc, "score": 0.5 * score}

    for doc, score in zip(sparse_docs, sparse_scores):
        key = doc.page_content
        if key in combined:
            combined[key]["score"] += 0.5 * score
        else:
            combined[key] = {"doc": doc, "score": 0.5 * score}

    merged = sorted(combined.values(), key=lambda x: x["score"], reverse=True)

    return [item["doc"] for item in merged[:top_k]]


def hybrid_rerank_search(query, top_k=6):

    merged_docs = hybrid_search(query, top_k)

    pairs = [(query, doc.page_content) for doc in merged_docs]
    rerank_scores = reranker.predict(pairs)

    reranked = sorted(
        zip(rerank_scores, merged_docs),
        key=lambda x: x[0],
        reverse=True
    )

    final_docs = [doc for score, doc in reranked[:3]]

    # Apply Lost-in-the-Middle mitigation
    final_docs = reorder_for_long_context(final_docs)

    return final_docs



# =========================================================
# 14. CUSTOM HYBRID RETRIEVER
# =========================================================

class HybridRetriever(BaseRetriever):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List:
        return hybrid_rerank_search(query)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List:
        return hybrid_rerank_search(query)


retriever = HybridRetriever()


# =========================================================
# 15. QUERY TYPE DETECTION (explicit mapping)
# =========================================================

QUERY_TYPE_MAP = {
    # Fact-Based
    "What is the maximum fine for a GDPR violation?": "fact",
    "What does GDPR say about the right to be forgotten?": "fact",
    "Under what conditions can personal data be transferred outside the EU?": "fact",
    "What is the role of a Data Protection Officer (DPO)?": "fact",
    "What constitutes a ‘data breach’ under GDPR?": "fact",
    "What are the lawful bases for processing personal data under GDPR?": "fact",
    "How long can organizations retain personal data under GDPR?": "fact",
    "What are the key rights of individuals regarding their personal data?": "fact",
    "What constitutes valid consent under GDPR?": "fact",
    "What obligations do data processors have under GDPR?": "fact",

    # Abstract
    "How does GDPR define data minimization?": "abstract",
    "Why is GDPR considered a landmark regulation in data privacy?": "abstract",
    "What are the key differences between explicit and implicit consent under GDPR?": "abstract",
    "How does GDPR affect AI-based data processing?": "abstract",

    # Reasoning
    "If a company stores customer data without informing users, which GDPR articles does it violate?": "reasoning",
    "Can a company use personal data without consent if they anonymize it?": "reasoning",
    "If a user deletes their account, does GDPR require their data to be erased immediately?": "reasoning",

    # Comparative
    "How does GDPR differ from the California Consumer Privacy Act (CCPA)?": "comparative",
    "How does GDPR handle children’s data protection compared to COPPA?": "comparative",
    "What are the key similarities between GDPR and Brazil’s LGPD?": "comparative",
}

def detect_query_type(query: str) -> str:
    return QUERY_TYPE_MAP.get(query.strip(), "fact")  # default fallback



# =========================================================
# 16. ANSWER GENERATION
# =========================================================

def answer_query(query):

    print("\n" + "=" * 60)
    print(f"\nQUESTION: {query}")
    print(f"QUERY TYPE: {detect_query_type(query).upper()}")
    print("=" * 60)

    # --- Retrieve ---
    retrieved_docs = hybrid_rerank_search(query)

    # --- Build context ---
    context = build_context(retrieved_docs)

    # --- Select prompt ---
    query_type      = detect_query_type(query)
    prompt_template = PROMPT_TEMPLATES[query_type]
    final_prompt    = prompt_template.format(
        context=context,
        query=query
    )

    # --- Generate ---
    try:
        answer = llm.invoke(final_prompt)

        print("\nANSWER:\n")
        print(answer)

        for idx, doc in enumerate(retrieved_docs):
            page    = doc.metadata.get("page_number", "?")
            chunk   = doc.metadata.get("chunk_id", "?")
            preview = doc.page_content[:60].replace("\n", " ")
            print(f"  {idx+1:<4} {str(page):>6}  {str(chunk):>7}  {preview}...")

    except Exception as e:
        print(f"\nERROR during generation: {e}")


# =========================================================
# 17. EVALUATION QUERIES
# =========================================================

queries = [
    # Fact-Based
    "What is the maximum fine for a GDPR violation?",
	"What does GDPR say about the right to be forgotten?",
	"Under what conditions can personal data be transferred outside the EU?",
	"What is the role of a Data Protection Officer (DPO)?",
	"What constitutes a ‘data breach’ under GDPR?",
	"What are the lawful bases for processing personal data under GDPR?",
	"How long can organizations retain personal data under GDPR?",
	"What are the key rights of individuals regarding their personal data?",
	"What constitutes valid consent under GDPR?",
    "What obligations do data processors have under GDPR?",
    # Abstract
    "How does GDPR define data minimization?",
    "Why is GDPR considered a landmark regulation in data privacy?",
	"What are the key differences between explicit and implicit consent under GDPR?",
	"How does GDPR affect AI-based data processing?",
    # Reasoning
    "If a company stores customer data without informing users, which GDPR articles does it violate?",
	"Can a company use personal data without consent if they anonymize it?",
	"If a user deletes their account, does GDPR require their data to be erased immediately?",
    # Comparative
    "How does GDPR differ from the California Consumer Privacy Act (CCPA)?",
	"How does GDPR handle children’s data protection compared to COPPA?",
	"What are the key similarities between GDPR and Brazil’s LGPD?"
]


# =========================================================
# 18. RUN FULL QA PIPELINE + WRITE OUTPUT (no sources)
# =========================================================

output_file = "qa_results.txt"

with open(output_file, "w", encoding="utf-8") as f:
    f.write("GDPR QA Pipeline Results\n")
    f.write("="*60 + "\n\n")

    for q in queries:
        print("\n" + "=" * 60)
        print("QUESTION:", q)
        print("=" * 60)

        # --- Retrieve ---
        retrieved_docs = hybrid_rerank_search(q)
        context = build_context(retrieved_docs)

        query_type      = detect_query_type(q)
        prompt_template = PROMPT_TEMPLATES[query_type]
        final_prompt    = prompt_template.format(context=context, query=q)

        try:
            answer = llm.invoke(final_prompt)

            # Print to console
            print("\nANSWER:\n", answer)

            # Write only Q & A to file
            f.write(f"QUESTION: {q}\n")
            f.write(f"ANSWER: {answer}\n")
            f.write("\n" + "-"*60 + "\n\n")

        except Exception as e:
            print(f"\nERROR during generation: {e}")
            f.write(f"QUESTION: {q}\nERROR: {e}\n\n")


# =========================================================
# 19. SEQUENTIAL FOLLOW-UP QUERY TEST
# =========================================================

print("\n" + "=" * 60)
print("SEQUENTIAL FOLLOW-UP QUERY TEST")
print("=" * 60)

follow_up_queries = [
    "If a company stores customer data without informing users, which GDPR articles does it violate?",
	"If a user deletes their account, does GDPR require their data to be erased immediately?",

]

for q in follow_up_queries:
    answer_query(q)
