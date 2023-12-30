

FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime
LABEL maintainer="m3e-server"
LABEL author="theone1006"

RUN mkdir /app

WORKDIR /app
ARG EMBEDD="moka-ai/m3e-base"

COPY requirements.txt /app
RUN pip install --no-cache-dir -r requirements.txt

ENV HF_HUB_ENABLE_HF_TRANSFER=1
RUN huggingface-cli download ${EMBEDD} --repo-type model --revision main

COPY util_feature.py /app
COPY m3e_server.py /app
EXPOSE 6800

ENV ENABLE_BEDDING = ${EMBEDD}

# python3 m3e_server.py moka-ai/m3e-base,moka-ai/m3e-small
CMD python3 m3e_server.py ${ENABLE_BEDDING}



