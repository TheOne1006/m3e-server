FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
LABEL maintainer="m3e-server"
LABEL author="theone1006"

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY util_feature.py /app
COPY m3e_server.py /app
EXPOSE 6800

VOLUME ~/.cache/huggingface/


CMD ["python3", "m3e_server.py", "--allow_models", "moka-ai/m3e-base"]

