# m3e server Docker

> feature
- 基于 SentenceTransformer 的服务, hugging face 上 sentence-transformers 模型都可以支持

## usage

1. 默认(使用 m3e-base 模型)
```bash
docker run -d -p 6800:6800 --gpus all --name m3e-server theone1006/m3e-server
```

2. 加载多个模型
启动时加载, 通过参数指定加载的模型, 例如: 

```bash
# 开启 moka-ai/m3e-base 和 moka-ai/m3e-large 模型
docker run -d -p 6800:6800 --name m3e-server theone1006/m3e-server:latest python m3e-server.py moka-ai/m3e-base,moka-ai/m3e-large
````

3. 加载自定义模型
构建时加载
```bash
docker run -d -p 6800:6800 --env  --name m3e-server theone1006/m3e-server:latest
```


## Test

```bash
curl -v --location --request POST 'http://127.0.0.1:6800/v1/embeddings' \
--header 'Content-Type: application/json' \
--data-raw '{
  "model": "m3e-base",
  "input": ["唱、跳、rap、篮球"]
}'
```


### Citation
    
```bibtex
  @software {Moka Massive Mixed Embedding,  
  author = {Wang Yuxin,Sun Qingxuan,He sicheng},  
  title = {M3E: Moka Massive Mixed Embedding Model},  
  year = {2023}
  }
```


### 代码参考

1. https://platform.openai.com/docs/api-reference/embeddings/create
2. https://github.com/byebyebruce/m3e-embed
3. https://hub.docker.com/layers/stawky/m3e-large-api
