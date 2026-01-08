from langchain_openai import OpenAIEmbeddings
from pymilvus import (
    connections, FieldSchema, CollectionSchema,
    DataType, Collection, utility
)

EMBED_DIM = 3072
MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"

connections.connect(
    alias="default",
    host=MILVUS_HOST,
    port=MILVUS_PORT
)

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-large"
)

def embed_chunks(chunks):
    return embedding_model.embed_documents(chunks)

def ensure_index(collection):
    if collection.has_index():
        return

    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "IP",
        "params": {"nlist": 1024}
    }

    collection.create_index(
        field_name="embedding",
        index_params=index_params
    )

def get_or_create_collection(collection_name: str):
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        ensure_index(collection)
        collection.load()
        return collection

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBED_DIM),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=10000),
        FieldSchema(name="domain", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="type", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="vendor", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="version", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="stage", dtype=DataType.VARCHAR, max_length=256),
    ]

    schema = CollectionSchema(fields, description="RAG PDF chunks")
    collection = Collection(collection_name, schema)
    ensure_index(collection)
    collection.load()
    return collection

def milvus_store(
    collection_name: str,
    chunks: list[str],
    domain: str,
    stage: str,
    type_: str,
    version: str,
    vendor: str,
    source: str
):
    collection = get_or_create_collection(collection_name)

    embeddings = embed_chunks(chunks)

    n = len(chunks)

    collection.insert([
        embeddings,
        chunks,
        [domain] * n,
        [type_] * n,
        [vendor] * n,
        [source] * n,
        [version] * n,
        [stage] * n
    ])

    collection.flush()
    print(f"Inserted {n} records into '{collection_name}'")

milvus_store("physical",
["Design Compiler® User Guide Version W-2024.09-SP3, January 20252 Copyright and Proprietary Information Notice © 2025 Synopsys, Inc. This Synopsys software and all associated documentation are proprietary to Synopsys, Inc. and may only be used pursuant to the terms and conditions of a written license agreement with Synopsys, Inc.",
" All other use, reproduction, modification, or distribution of the Synopsys software or the associated documentation is strictly prohibited."],
"physical","tool","synposys","dcug","w-293","sythesis")