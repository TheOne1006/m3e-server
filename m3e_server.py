import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
import torch
from util_feature import expand_features
from config import ALLOW_MODELS, EXPORT_DIM, NORMALIZE

shortModel = {
    "m3e-base": "moka-ai/m3e-base",
    "m3e-large": "moka-ai/m3e-large",
    "m3e-small": "moka-ai/m3e-small",
}

enableModels = []
cacheModels = {}

app = FastAPI(docs_url='/')


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


def text_embeddings(model: str, inputs: list[str], export_dims=0) -> dict:
    full_model = shortModel.get(model, model)
    
    # check enable model
    if full_model not in enableModels:
        raise ValueError(f"model: {model} not supported, enable models: {enableModels}")
    
    # create instance And cache
    if full_model in cacheModels:
        embeddingModel = cacheModels[full_model]
    else:
        embeddingModel = SentenceTransformer(full_model)
        cacheModels[full_model] = embeddingModel
    embeddings = embeddingModel.encode(inputs)
    
    if export_dims > 0:
        embeddings = [expand_features(item, export_dims) if len(item) < export_dims else item for item in embeddings]
    
    if NORMALIZE:
        embeddings = [embedding / np.linalg.norm(embedding) for embedding in embeddings]
    
    embeddings = [embedding.tolist() for embedding in embeddings]
    prompt_tokens = sum(len(text.split()) for text in inputs)
    total_tokens = sum(num_tokens_from_string(text) for text in inputs)
    
    response = {
        "data": [
            {
                "embedding": embedding,
                "index": index,
                "object": "embedding"
            } for index, embedding in enumerate(embeddings)
        ],
        "model": model,
        "object": "list",
        "usage": {
            "prompt_tokens": prompt_tokens,
            "total_tokens": total_tokens,
        }
    }
    
    return response


class EmbeddingBody(BaseModel):
    model: str = 'm3e-base',
    input: list[str] | str = None,


@app.post('/v1/embeddings')
def route_text_embeddings(item: EmbeddingBody):
    model = item.model
    inputs = item.input
    
    # 兼容 string
    if isinstance(inputs, str):
        inputs = [inputs]
    
    response = text_embeddings(model, inputs, EXPORT_DIM)
    
    return response


def check_server():
    print("checking models: ", ALLOW_MODELS)
    for model_name in ALLOW_MODELS:
        try:
            _ = SentenceTransformer(model_name)
            enableModels.append(model_name)
        except Exception as e:
            print(f"model: {model_name} not supported")
            continue
    
    print("finished, enabled models: ", enableModels)
    
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")
    
    if EXPORT_DIM > 0:
        print(f"EXPORT_DIM: {EXPORT_DIM}")

check_server()

if __name__ == '__main__':
    uvicorn.run(app=app)
