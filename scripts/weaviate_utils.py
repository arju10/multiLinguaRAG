import os
import weaviate
from dotenv import load_dotenv
from pymongo import MongoClient
from openai import OpenAI

load_dotenv()

# MongoDB
mongo_client = MongoClient(os.getenv("MONGO_URI"))
db = mongo_client["multilinguarag"]
collection = db["chunks"]

# Weaviate client
weaviate_client = weaviate.Client(
    url=os.getenv("WEAVIATE_URL")
)

# OpenAI client
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---- STEP 1: Schema ----
class_name = "BanglaChunk"

if not weaviate_client.schema.contains({"class": class_name}):
    schema = {
        "classes": [
            {
                "class": class_name,
                "vectorizer": "none",
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"]
                    }
                ]
            }
        ]
    }
    weaviate_client.schema.create(schema)
    print("✅ Weaviate Schema Created!")
else:
    print("✅ Weaviate Schema Already Exists!")

# ---- STEP 2: Get Chunks ----
doc = collection.find_one({"book_name": "HSC26 Bangla 1st Paper"})
chunks = doc["semantic_chunks"]

# ---- STEP 3: Embed & Push ----
for i, chunk in enumerate(chunks):
    response = openai_client.embeddings.create(
        model="text-embedding-3-large",
        input=chunk
    )
    embedding = response.data[0].embedding

    data_object = {
        "content": chunk
    }

    weaviate_client.data_object.create(
        class_name=class_name,
        properties=data_object,
        vector=embedding
    )

    if (i + 1) % 10 == 0:
        print(f"✅ Uploaded {i+1}/{len(chunks)}")

print("🎉 All chunks uploaded to Weaviate!")

mongo_client.close()
