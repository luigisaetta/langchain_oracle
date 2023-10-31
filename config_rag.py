# configurations for the RAG

TIMEOUT = 30

# to divide docs in chunks
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50

# number of docs to return from retriever
MAX_DOCS_RETRIEVED = 5

# book to use for augmentation
BOOK1 = "./oracle-database-23c-new-features-guide.pdf"
BOOK2 = "./database-concepts.pdf"

BOOK_LIST = [BOOK1, BOOK2]

#
# Vector Store (Chrome or FAISS)
#
# VECTOR_STORE_NAME = "FAISS"
VECTOR_STORE_NAME = "CHROME"

#
# LLM Config
#
# LLM_TYPE = "COHERE"
LLM_TYPE = "OCI"

# max token returned from LLM for single query
MAX_TOKENS = 1500

TEMPERATURE = 0


#
# OCI GenAI configs
#

# endpoint for OCI GenAI
ENDPOINT = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

# type of Embedding Model. The choice has been parametrized
# Local means HF
EMBED_TYPE = "LOCAL"
# see: https://huggingface.co/spaces/mteb/leaderboard
# see also: https://github.com/FlagOpen/FlagEmbedding
# base seems to work better than small
EMBED_HF_MODEL_NAME = "BAAI/bge-base-en-v1.5"
# EMBED_HF_MODEL_NAME = "BAAI/bge-small-en-v1.5"
# EMBED_HF_MODEL_NAME = "BAAI/bge-large-en-v1.5"

# Cohere means the embed model from Cohere site API
# EMBED_TYPE = "COHERE"
