# configurations for the RAG

# to divide docs in chunks
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50

# number of docs to return from retriever
MAX_DOCS_RETRIEVED = 5

# book to use for augmentation
BOOK = "./oracle-database-23c-new-features-guide.pdf"
# BOOK = "./feynman_vol1.pdf"

# endpoint for OCI GenAI
ENDPOINT = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

# max token returned from LLM for single query
MAX_TOKENS = 1500

# type of Embedding Model. The choice has been parametrized
# Local means HF BAAI/bge-base-en-v1.5
EMBED_TYPE = "LOCAL"
# see: https://huggingface.co/spaces/mteb/leaderboard
# EMBED_HF_MODEL_NAME = "BAAI/bge-base-en-v1.5"
EMBED_HF_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Cohere means the embed model from Cohere site API
# EMBED_TYPE = "COHERE"
