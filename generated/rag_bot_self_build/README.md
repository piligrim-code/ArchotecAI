# rag_bot_self_build — generated organ

COGOS was given the goal "build a RAG bot" and designed the system into four organs
(capabilities). Of those four, only **one** produced real, working code:

| organ | status |
|---|---|
| `similarity_searcher` (cosine top-k retrieval) | **generated, real code** — kept here as-is |
| `document_chunker` | failed auto-validation (`layer=tests`) — no usable code produced |
| `embedding_computer` | failed auto-validation — no usable code produced |
| `llm_responder` | no code produced |

`similarity_searcher.py` in this directory is exactly what COGOS produced — unedited. It's a
correct, if unremarkable, cosine-similarity top-k lookup over pre-computed vectors (sklearn +
numpy). The other three organs aren't included because there's nothing generated to show for
them.
