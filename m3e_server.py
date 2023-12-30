import os.path
from waitress import serve
from flask import Flask, jsonify, request
import numpy as np
from sentence_transformers import SentenceTransformer
import tiktoken
import torch
from util_feature import expand_features


shortModel = {
    "m3e-base": "moka-ai/m3e-base",
    "m3e-large": "moka-ai/m3e-large",
    "m3e-small": "moka-ai/m3e-small",
}


enableModels = []
cacheModels = {}

app = Flask(__name__)


def num_tokens_from_string(string: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding('cl100k_base')
    num_tokens = len(encoding.encode(string))
    return num_tokens


EXPORT_DIM = os.environ.get('EXPORT_DIM', '0')
try:
    EXPORT_DIM = int(EXPORT_DIM)
except ValueError as e:
    print(f"EXPORT_DIM :{EXPORT_DIM} is Error")
    EXPORT_DIM = 0


def text_embeddings(model: str, inputs: list[str], export_dims=0) -> dict:
    full_model = shortModel.get(model, model)
    
    # check enable model
    if full_model not in enableModels:
        raise ValueError(f"model: {model} not supported")
    
    # create instance And cache
    if full_model in cacheModels:
        embeddingModel = cacheModels[full_model]
    else:
        embeddingModel = SentenceTransformer(full_model)
        cacheModels[full_model] = embeddingModel
        
    embeddings = embeddingModel.encode(inputs)
    
    if export_dims > 0:
        embeddings = [expand_features(item, export_dims) if len(item) < export_dims else item for item in embeddings]
    
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


@app.route('/v1/embeddings', methods=['POST'])
def route_text_embeddings():
    data = request.get_json()
    model = data.get('model', 'm3e-base')
    inputs = data.get('input', [])
    
    try:
        response = text_embeddings(model, inputs, EXPORT_DIM)
    except ValueError as e:
        err_msg = {
            "error": str(e)
        }
        return jsonify(err_msg), 400

    return jsonify(response)


if __name__ == '__main__':
    import sys
    allow_models = sys.argv[2]
    allow_models_arr = allow_models.split(',')
    
    for model_name in allow_models_arr:
        try:
            _ = SentenceTransformer(model_name)
            enableModels.append(model_name)
        except Exception as e:
            print(f"model: {model_name} not supported")
            continue
    
    if torch.cuda.is_available():
        print("cuda is available")
    else:
        print("cuda is not available")
    
    print(f"allow_models: {allow_models}")
    print(f"app start on port: 0.0.0.0:6800")
    serve(app, host="0.0.0.0", port=6800)