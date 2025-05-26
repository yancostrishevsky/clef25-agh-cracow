FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y openjdk-11-jdk git wget tzdata build-essential g++ && \
    ln -fs /usr/share/zoneinfo/Europe/Warsaw /etc/localtime && \
    dpkg-reconfigure -f noninteractive tzdata && \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
ENV PATH="$JAVA_HOME/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install faiss-cpu
RUN pip install -r requirements.txt
RUN pip install --upgrade torch

RUN pip install pyserini==0.20.0

# -- TU DODAJ TO -- #
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('intfloat/e5-base-v2')" && \
    python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')"
# ----------------- #

COPY . /app
WORKDIR /app

ENTRYPOINT ["/bin/bash", "run_pipeline.sh"]
