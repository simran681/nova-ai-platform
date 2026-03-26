"""
NOVA AI Platform — RAG Module (Task 3)
Importable RAG pipeline used by both Task 3 notebook and Task 5 multi-agent platform.

Pipeline:
  Query → BM25 (sparse) + ChromaDB (dense) → RRF fusion → cross-encoder rerank → LLM answer

Usage:
    from rag_module import NOVARAGPipeline
    rag = NOVARAGPipeline()
    rag.build_index(documents)
    result = rag.query("Does the Glow Serum contain niacinamide?")
"""

import json
import re
import time
from typing import Optional
from dataclasses import dataclass, field


@dataclass
class Document:
    """A document in the NOVA knowledge base."""
    doc_id: str
    content: str
    metadata: dict = field(default_factory=dict)


@dataclass
class RetrievedChunk:
    """A retrieved and scored document chunk."""
    doc_id: str
    content: str
    metadata: dict
    bm25_score: float = 0.0
    dense_score: float = 0.0
    rrf_score: float = 0.0
    rerank_score: float = 0.0


@dataclass
class RAGResult:
    """Full result from a RAG query."""
    query: str
    answer: str
    retrieved_chunks: list
    latency_ms: float
    model_used: str
    citations: list


class NOVARAGPipeline:
    """
    NOVA Product Knowledge RAG Pipeline.

    Implements hybrid search (BM25 + ChromaDB) with cross-encoder reranking.
    Designed to run on Google Colab Free Tier (CPU or T4 GPU).
    """

    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        chroma_collection: str = "nova_products",
        chroma_path: str = "./chroma_db",
        llm_client=None,
        llm_model: str = "llama-3.1-8b-instant",
        top_k_retrieval: int = 10,
        top_k_final: int = 3,
        rrf_k: int = 60,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ):
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        self.chroma_collection_name = chroma_collection
        self.chroma_path = chroma_path
        self.llm_client = llm_client
        self.llm_model = llm_model
        self.top_k_retrieval = top_k_retrieval
        self.top_k_final = top_k_final
        self.rrf_k = rrf_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Lazy-loaded components
        self._embedder = None
        self._reranker = None
        self._chroma_client = None
        self._chroma_collection = None
        self._bm25_index = None
        self._bm25_corpus = []
        self._documents = []
        self._is_built = False

    def _get_embedder(self):
        """Lazy-load sentence transformer embedder."""
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            print(f"  Loading embedding model: {self.embedding_model_name}")
            self._embedder = SentenceTransformer(self.embedding_model_name)
        return self._embedder

    def _get_reranker(self):
        """Lazy-load cross-encoder reranker."""
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            print(f"  Loading reranker: {self.reranker_model_name}")
            self._reranker = CrossEncoder(self.reranker_model_name)
        return self._reranker

    def _get_chroma(self):
        """Lazy-load ChromaDB client."""
        if self._chroma_client is None:
            import chromadb
            self._chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        return self._chroma_client

    @staticmethod
    def _tokenize(text: str) -> list:
        """Simple whitespace + punctuation tokenizer for BM25."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        return text.split()

    def chunk_document(self, doc: Document) -> list:
        """
        Split a document into overlapping chunks of ~chunk_size words.
        Returns list of (chunk_text, chunk_metadata).
        """
        words = doc.content.split()
        chunks = []
        start = 0

        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            chunk_meta = {
                **doc.metadata,
                "doc_id": doc.doc_id,
                "chunk_index": len(chunks),
                "char_start": start,
                "char_end": end
            }
            chunks.append((chunk_text, chunk_meta))
            start += self.chunk_size - self.chunk_overlap

        return chunks

    def build_index(self, documents: list, reset: bool = False):
        """
        Build the hybrid search index from a list of Document objects.

        Args:
            documents: List of Document objects
            reset: If True, delete existing ChromaDB collection and rebuild
        """
        print(f"\nBuilding NOVA RAG index from {len(documents)} documents...")
        t0 = time.perf_counter()

        # Chunk all documents
        all_chunks = []
        chunk_ids = []
        chunk_texts = []
        chunk_metas = []

        for doc in documents:
            chunks = self.chunk_document(doc)
            for i, (text, meta) in enumerate(chunks):
                cid = f"{doc.doc_id}_chunk_{i}"
                all_chunks.append((cid, text, meta))
                chunk_ids.append(cid)
                chunk_texts.append(text)
                chunk_metas.append(meta)

        print(f"  Chunked {len(documents)} docs → {len(all_chunks)} chunks")
        self._documents = documents
        self._bm25_corpus = [self._tokenize(t) for t in chunk_texts]

        # Build BM25 index
        from rank_bm25 import BM25Okapi
        self._bm25_index = BM25Okapi(self._bm25_corpus)
        self._chunk_texts = chunk_texts
        self._chunk_ids = chunk_ids
        self._chunk_metas = chunk_metas
        print(f"  BM25 index built ({len(self._bm25_corpus)} entries)")

        # Build ChromaDB dense index
        chroma = self._get_chroma()
        if reset:
            try:
                chroma.delete_collection(self.chroma_collection_name)
            except Exception:
                pass

        try:
            self._chroma_collection = chroma.get_collection(self.chroma_collection_name)
            if self._chroma_collection.count() > 0 and not reset:
                print(f"  ChromaDB collection loaded ({self._chroma_collection.count()} entries)")
                self._is_built = True
                return
        except Exception:
            pass

        self._chroma_collection = chroma.create_collection(
            name=self.chroma_collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        # Embed in batches (avoid OOM on Colab free)
        embedder = self._get_embedder()
        batch_size = 32
        all_embeddings = []

        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            embeddings = embedder.encode(batch, show_progress_bar=False).tolist()
            all_embeddings.extend(embeddings)
            if (i // batch_size) % 5 == 0:
                print(f"  Embedding progress: {min(i + batch_size, len(chunk_texts))}/{len(chunk_texts)}")

        # Add to ChromaDB
        self._chroma_collection.add(
            ids=chunk_ids,
            embeddings=all_embeddings,
            documents=chunk_texts,
            metadatas=chunk_metas
        )

        elapsed = (time.perf_counter() - t0) * 1000
        print(f"  ChromaDB index built ({len(chunk_ids)} embeddings)")
        print(f"  Total index build time: {elapsed:.0f}ms")
        self._is_built = True

    def _bm25_search(self, query: str) -> list:
        """Sparse BM25 retrieval. Returns list of (chunk_idx, score) sorted desc."""
        tokens = self._tokenize(query)
        scores = self._bm25_index.get_scores(tokens)
        ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return ranked[:self.top_k_retrieval]

    def _dense_search(self, query: str) -> list:
        """Dense ChromaDB semantic retrieval. Returns list of (chunk_id, score)."""
        embedder = self._get_embedder()
        query_embedding = embedder.encode(query).tolist()

        results = self._chroma_collection.query(
            query_embeddings=[query_embedding],
            n_results=self.top_k_retrieval,
            include=["documents", "metadatas", "distances"]
        )

        chunk_ids = results["ids"][0]
        distances = results["distances"][0]

        # Convert cosine distance to similarity score (0-1)
        similarities = [1 - d for d in distances]

        # Map chunk_id back to index
        id_to_idx = {cid: i for i, cid in enumerate(self._chunk_ids)}
        return [(id_to_idx.get(cid, -1), sim)
                for cid, sim in zip(chunk_ids, similarities)
                if cid in id_to_idx]

    def _rrf_fusion(self, bm25_ranked: list, dense_ranked: list) -> list:
        """
        Reciprocal Rank Fusion: combines BM25 and dense rankings.
        RRF score = sum(1 / (k + rank)) for each retriever.
        """
        rrf_scores = {}

        for rank, (idx, _) in enumerate(bm25_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank + 1)

        for rank, (idx, _) in enumerate(dense_ranked):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (self.rrf_k + rank + 1)

        # Sort by RRF score descending
        return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    def _rerank(self, query: str, candidates: list) -> list:
        """
        Cross-encoder reranking of top candidates.
        Returns candidates sorted by rerank score descending.
        """
        reranker = self._get_reranker()

        pairs = [(query, self._chunk_texts[idx]) for idx, _ in candidates]
        rerank_scores = reranker.predict(pairs)

        reranked = sorted(
            zip(candidates, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [(idx, rrf_score, rerank_score)
                for (idx, rrf_score), rerank_score in reranked]

    def retrieve(self, query: str) -> list:
        """
        Hybrid retrieval: BM25 + dense → RRF → rerank → top-k chunks.

        Returns list of RetrievedChunk objects sorted by relevance.
        """
        if not self._is_built:
            raise RuntimeError("Index not built. Call build_index() first.")

        # Step 1: Sparse + dense retrieval
        bm25_results = self._bm25_search(query)
        dense_results = self._dense_search(query)

        # Collect BM25 scores for logging
        bm25_score_map = {idx: score for idx, score in bm25_results}
        dense_score_map = {idx: score for idx, score in dense_results}

        # Step 2: RRF fusion
        fused = self._rrf_fusion(bm25_results, dense_results)
        top_candidates = fused[:self.top_k_retrieval]

        # Step 3: Cross-encoder reranking
        reranked = self._rerank(query, top_candidates)
        final = reranked[:self.top_k_final]

        # Build result objects
        chunks = []
        for idx, rrf_score, rerank_score in final:
            chunks.append(RetrievedChunk(
                doc_id=self._chunk_ids[idx],
                content=self._chunk_texts[idx],
                metadata=self._chunk_metas[idx],
                bm25_score=bm25_score_map.get(idx, 0.0),
                dense_score=dense_score_map.get(idx, 0.0),
                rrf_score=rrf_score,
                rerank_score=float(rerank_score)
            ))

        return chunks

    def query(self, question: str, return_raw: bool = False) -> RAGResult:
        """
        Full RAG pipeline: retrieve + generate answer.

        Args:
            question: Customer question
            return_raw: If True, skip LLM generation and return chunks only

        Returns:
            RAGResult with answer, retrieved chunks, and metadata
        """
        t0 = time.perf_counter()

        # Retrieve relevant chunks
        chunks = self.retrieve(question)

        if return_raw or self.llm_client is None:
            answer = "\n\n".join([f"[{c.doc_id}]\n{c.content}" for c in chunks])
            return RAGResult(
                query=question,
                answer=answer,
                retrieved_chunks=chunks,
                latency_ms=(time.perf_counter() - t0) * 1000,
                model_used="retrieval_only",
                citations=[c.doc_id for c in chunks]
            )

        # Build context for LLM
        context = self._build_context(chunks)

        # Generate answer with LLM
        answer = self._generate_answer(question, context)

        latency = (time.perf_counter() - t0) * 1000
        return RAGResult(
            query=question,
            answer=answer,
            retrieved_chunks=chunks,
            latency_ms=latency,
            model_used=self.llm_model,
            citations=[c.metadata.get("source", c.doc_id) for c in chunks]
        )

    def _build_context(self, chunks: list) -> str:
        """Format retrieved chunks into LLM context."""
        parts = []
        for i, chunk in enumerate(chunks, 1):
            source = chunk.metadata.get("product_name", chunk.metadata.get("source", chunk.doc_id))
            parts.append(f"[Source {i}: {source}]\n{chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _generate_answer(self, question: str, context: str) -> str:
        """Generate grounded answer using the LLM."""
        prompt = f"""You are NOVA's product knowledge assistant. Answer the customer's question using ONLY the information provided in the context below.

If the context does not contain enough information to answer fully, say so clearly and suggest the customer contact support.

Always cite which source(s) you used. Keep your answer concise and in NOVA's warm, friendly tone.

CONTEXT:
{context}

CUSTOMER QUESTION: {question}

ANSWER:"""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=400
        )
        return response.choices[0].message.content.strip()

    def get_index_stats(self) -> dict:
        """Return statistics about the current index."""
        return {
            "is_built": self._is_built,
            "num_documents": len(self._documents),
            "num_chunks": len(self._chunk_texts) if hasattr(self, '_chunk_texts') else 0,
            "embedding_model": self.embedding_model_name,
            "reranker_model": self.reranker_model_name,
            "top_k_retrieval": self.top_k_retrieval,
            "top_k_final": self.top_k_final,
            "chroma_collection": self.chroma_collection_name
        }


# ── Knowledge Base Builder ─────────────────────────────────────────────────────

def build_nova_knowledge_base(db_path: str = "nova_mock_db.json") -> list:
    """
    Build NOVA's product knowledge base from the mock database.
    Returns a list of Document objects ready for indexing.
    """
    with open(db_path) as f:
        db = json.load(f)

    documents = []

    # 1. Product documents (with full ingredient lists)
    for product in db["products"]:
        content_parts = [
            f"Product: {product['name']}",
            f"Category: {product['category']}",
            f"Price: ${product['price']}",
            f"Description: {product.get('description', '')}",
        ]

        if product.get("ingredients"):
            content_parts.append(f"Ingredients: {', '.join(product['ingredients'])}")

        if product.get("skin_types"):
            content_parts.append(f"Suitable for skin types: {', '.join(product['skin_types'])}")

        if product.get("concerns"):
            content_parts.append(f"Addresses concerns: {', '.join(product['concerns'])}")

        if product.get("sizes"):
            content_parts.append(f"Available sizes: {', '.join(str(s) for s in product['sizes'])}")

        if product.get("material"):
            content_parts.append(f"Material: {product['material']}")

        if product.get("volume_ml"):
            content_parts.append(f"Volume: {product['volume_ml']}ml")

        if product.get("spf"):
            content_parts.append(f"SPF: {product['spf']}")

        content_parts.append(
            f"Rating: {product.get('rating', 'N/A')}/5 "
            f"({product.get('review_count', 0)} reviews)"
        )
        content_parts.append("All NOVA products are cruelty-free. "
                             "Many are vegan — check the product page for details.")

        content = "\n".join(content_parts)
        documents.append(Document(
            doc_id=product["product_id"],
            content=content,
            metadata={
                "source": "product_catalog",
                "product_id": product["product_id"],
                "product_name": product["name"],
                "category": product["category"],
                "doc_type": "product"
            }
        ))

    # 2. FAQ documents
    for faq in db.get("faqs", []):
        content = f"Q: {faq['question']}\nA: {faq['answer']}"
        documents.append(Document(
            doc_id=faq["id"],
            content=content,
            metadata={
                "source": "faq",
                "faq_id": faq["id"],
                "category": faq["category"],
                "doc_type": "faq"
            }
        ))

    # 3. Sizing guide (appended as a dedicated document)
    sizing_guide = """NOVA Sizing Guide

CLOTHING SIZES (UK/EU/US):
- XS: UK 6-8 | EU 34-36 | US 2-4 | Bust 76-80cm | Waist 56-60cm | Hips 81-85cm
- S:  UK 8-10 | EU 36-38 | US 4-6 | Bust 80-84cm | Waist 60-64cm | Hips 85-89cm
- M:  UK 10-12 | EU 38-40 | US 6-8 | Bust 84-88cm | Waist 64-68cm | Hips 89-93cm
- L:  UK 12-14 | EU 40-42 | US 8-10 | Bust 88-92cm | Waist 68-72cm | Hips 93-97cm
- XL: UK 14-16 | EU 42-44 | US 10-12 | Bust 92-100cm | Waist 72-80cm | Hips 97-105cm
- XXL: UK 16-18 | EU 44-46 | US 12-14 | Bust 100-108cm | Waist 80-88cm | Hips 105-113cm

HOW TO MEASURE:
- Bust: Measure around the fullest part of your chest, keeping tape parallel to floor.
- Waist: Measure around your natural waistline (narrowest point).
- Hips: Measure around the fullest part of your hips/bottom.
- If between sizes, we recommend sizing UP for comfort.

FOOTWEAR SIZES (EU/UK/US):
- EU 35 | UK 2.5 | US 5 | Foot length: 22.5cm
- EU 36 | UK 3.5 | US 5.5 | Foot length: 23cm
- EU 37 | UK 4 | US 6.5 | Foot length: 23.5cm
- EU 38 | UK 5 | US 7 | Foot length: 24.5cm
- EU 39 | UK 6 | US 8 | Foot length: 25cm
- EU 40 | UK 6.5 | US 9 | Foot length: 25.5cm
- EU 41 | UK 7.5 | US 9.5 | Foot length: 26cm
- EU 42 | UK 8 | US 10 | Foot length: 26.5cm

HOW TO MEASURE YOUR FOOT:
Stand on paper, trace your foot outline, measure from heel to longest toe.
For wide feet, size up. For narrow feet, size down or choose adjustable styles."""

    documents.append(Document(
        doc_id="GUIDE-SIZING",
        content=sizing_guide,
        metadata={
            "source": "sizing_guide",
            "doc_type": "guide",
            "category": "sizing"
        }
    ))

    # 4. Ingredient safety guide
    ingredient_guide = """NOVA Ingredient Safety & Compatibility Guide

INGREDIENTS TO AVOID DURING PREGNANCY:
- Retinol (Vitamin A) — avoid high concentrations; use Bakuchiol as a safe alternative
- Salicylic Acid — avoid in large amounts; small amounts (≤2%) generally considered safe
- Hydroquinone — not recommended during pregnancy
- Chemical sunscreen filters (Oxybenzone) — opt for mineral SPF (Zinc Oxide, Titanium Dioxide)

INGREDIENTS SAFE DURING PREGNANCY:
- Hyaluronic Acid — hydrating and safe
- Vitamin C (Ascorbic Acid) — antioxidant, safe
- Niacinamide — safe in standard concentrations
- Glycerin — safe humectant
- Ceramides — barrier-repairing, safe
- Bakuchiol — plant-based retinol alternative, safe
- Zinc Oxide / Titanium Dioxide (mineral SPF) — safe

COMMON ALLERGENS TO WATCH FOR:
- Fragrance (Parfum) — most common cause of contact dermatitis
- Lanolin — wool-derived, avoid if wool-allergic
- Beeswax — avoid if vegan or bee-product allergic
- Nickel (in accessories) — can cause metal allergies

VEGAN vs CRUELTY-FREE:
- All NOVA products are cruelty-free (Leaping Bunny certified)
- Vegan products exclude: Beeswax, Lanolin, Carmine, Silk/Sericin, Honey, Collagen (animal)
- Non-vegan ingredients sometimes present: Beeswax (lip products), Lanolin (hair masks)

SKIN TYPE INGREDIENT GUIDE:
Oily/Acne-prone: Niacinamide, Salicylic Acid, BHA, Zinc, lightweight Hyaluronic Acid
Dry skin: Hyaluronic Acid, Ceramides, Squalane, Glycerin, heavier oils
Sensitive skin: Avoid: Fragrance, Alcohol (drying), AHA/BHA (in high %), Retinol
               Use: Ceramides, Centella Asiatica, Aloe Vera, minimal ingredients
Combination: Lightweight hydrators for all zones; mattifiers for T-zone"""

    documents.append(Document(
        doc_id="GUIDE-INGREDIENTS",
        content=ingredient_guide,
        metadata={
            "source": "ingredient_guide",
            "doc_type": "guide",
            "category": "product_safety"
        }
    ))

    print(f"Built knowledge base: {len(documents)} documents "
          f"({sum(1 for d in documents if d.metadata.get('doc_type') == 'product')} products, "
          f"{sum(1 for d in documents if d.metadata.get('doc_type') == 'faq')} FAQs, "
          f"{sum(1 for d in documents if d.metadata.get('doc_type') == 'guide')} guides)")

    return documents
