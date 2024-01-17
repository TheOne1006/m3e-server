FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
LABEL maintainer="m3e-server"
LABEL author="theone1006"

RUN mkdir /app

WORKDIR /app

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

COPY util_feature.py /app
COPY config.py /app
COPY m3e_server.py /app

VOLUME ~/.cache/huggingface/
# support chinese
ENV LANG=C.UTF-8
# disable download model
#ENV TRANSFORMERS_OFFLINE=1

CMD ["uvicorn", "m3e_server:app", "--host", "0.0.0.0", "--port", "7860"]

