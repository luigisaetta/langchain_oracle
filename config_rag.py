# configurations for the RAG

# to divide docs in chunks
CHUNK_SIZE = 800
CHUNK_OVERLAP = 50

BOOK = "./oracle-database-23c-new-features-guide.pdf"

# endpoint for OCI GenAI
ENDPOINT = "https://generativeai.aiservice.us-chicago-1.oci.oraclecloud.com"

# max token returned form LLM for single query
MAX_TOKENS = 1500