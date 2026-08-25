# Base model for this lab.
#
# Must be a JumpStart model that (a) supports serverless customization and
# (b) can be served by the SageMaker LMI/DJL container and imported into
# Amazon Bedrock Custom Model Import.
#
# This branch trains Nemotron 3 Nano, a reasoning model. Notebook 1 prefixes every
# training completion with an empty reasoning block so the model answers instead of
# reasoning past the output budget. See nemotron_support.py for why.
BASE_MODEL_ID = "huggingface-reasoning-nvidia-nemotron-3-nano-30b-a3b-bf16"

# Fixed dataset / resource names used across the notebooks
DATASET_PREFIX = "contractnli-nda-review-nothink"
