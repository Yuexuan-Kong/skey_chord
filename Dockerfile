FROM registry.deez.re/research/python-audio-gpu-11-2:latest

# libgoogle-perftools: cf. https://stackoverflow.com/questions/57180479/tensorflow-serving-oomkilled-or-evicted-pod-with-kubernetes
# poetry: wait until poetry is installed in python-audio img
ENV POETRY_CACHE_DIR=/var/cache
ENV POETRY_VIRTUALENVS_CREATE=false
RUN apt-get update \
    && apt-get install -y ffmpeg curl libgoogle-perftools4 \
    && mkdir -p /var/probes && touch /var/probes/ready \
    && chmod 777 /var/cache \
    && pip install --upgrade poetry

# GSUTIL SDK
# Downloading gcloud package
RUN curl https://dl.google.com/dl/cloudsdk/release/google-cloud-sdk.tar.gz > /tmp/google-cloud-sdk.tar.gz

# Installing the package
RUN mkdir -p /usr/local/gcloud \
    && tar -C /usr/local/gcloud -xvf /tmp/google-cloud-sdk.tar.gz \
    && /usr/local/gcloud/google-cloud-sdk/install.sh

# Adding the package path to local
ENV PATH $PATH:/usr/local/gcloud/google-cloud-sdk/bin

RUN gcloud components update

ENV LD_PRELOAD /usr/lib/x86_64-linux-gnu/libtcmalloc.so.4

WORKDIR /workspace
COPY pyproject.toml .
COPY poetry.lock .

# Install dependencies
RUN poetry install --no-root

COPY gcloud.sh .
COPY .flake8 .

USER deezer