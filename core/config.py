import os

from dotenv import load_dotenv

load_dotenv(override=True)

ENVIRONMENT = os.getenv("ENVIRONMENT")

if ENVIRONMENT == "production":
    DB_USER = os.getenv("PROD_DB_USER")
    DB_PASSWORD = os.getenv("PROD_DB_PASSWORD")
    DB_NAME = os.getenv("PROD_DB_NAME")
    DB_HOST = os.getenv("PROD_DB_HOST")
    DB_PORT = os.getenv("PROD_DB_PORT")
    SSL_MODE = os.getenv("SSL_MODE")
    DB_CA_PATH = os.getenv("DB_CA_PATH")

    DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode={SSL_MODE}&sslrootcert={DB_CA_PATH}"
else:
    DB_USER = os.getenv("DB_USER")
    DB_PASSWORD = os.getenv("DB_PASSWORD")
    DB_NAME = os.getenv("DB_NAME")
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT")
    DATABASE_URI = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

API_NAME = os.getenv("API_NAME")
API_VERSION = os.getenv("API_VERSION", "v1")
