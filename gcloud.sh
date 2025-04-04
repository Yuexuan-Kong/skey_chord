gcloud auth application-default login
gcloud config set project dzr-research-dev
mkdir /tmp/audio
gcsfuse --implicit-dirs media-buckets-po-mp3_128 /tmp/audio
