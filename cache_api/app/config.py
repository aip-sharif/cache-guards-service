import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------
# Database
# ---------------------------------------------------------
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")
POSTGRES_DB = os.getenv("POSTGRES_DB")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT")

DATABASE_URL = (
    f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}"
)
#DATABASE_URL = os.getenv("DATABASE_URL")

PUBLIC_KEY= os.getenv("CERT")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
CASSDOOR_ENDPOINT = os.getenv("CASSDOOR_ENDPOINT")
APPLICATION_NAME = os.getenv("APPLICATION_NAME")
ORGANZATION_NAME = os.getenv("ORGANZATION_NAME")
CALLBACK_URL = os.getenv("CALLBACK_URL")
STATE_SECRET = os.getenv("STATE_SECRET")

EXTERNAL_API_URL =os.getenv("EXTERNAL_API_URL")
SC_GATEWAY_ADMIN_KEY = os.getenv("SC_GATEWAY_ADMIN_KEY")
SC_LLM_BASE_URL=os.getenv("SC_LLM_BASE_URL")               
SC_EMBED_BASE_URL=os.getenv("SC_LLM_BASE_URL") 

SC_APP_SERVICE_KEY = os.getenv("SC_APP_SERVICE_KEY")
SC_APP_SERVICE_KEY_HEADER = os.getenv("SC_APP_SERVICE_KEY_HEADER", "X-Service-Key")