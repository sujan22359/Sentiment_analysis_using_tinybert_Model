import streamlit as st
import os
import torch
from transformers import pipeline
import boto3

# AWS S3 config
bucket_name = "sentiment-analysis-tinybert-sujan"
s3_prefix = "ml-models/tinybert-sentiment-analysis/"
local_path = "tinybert-sentiment-analysis"

# Streamlit Cloud will load AWS keys from secrets
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_DEFAULT_REGION", "ap-south-1")
)

def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator("list_objects_v2")
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for obj in result['Contents']:
                s3_key = obj['Key']
                rel_path = os.path.relpath(s3_key, s3_prefix)
                
                # skip prefix root
                if rel_path == ".":
                    continue
                
                local_file = os.path.join(local_path, rel_path)
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)

st.title("Sentiment Analysis using TinyBERT Model")

# --------------------------
# DOWNLOAD MODEL BUTTON
# --------------------------
if st.button("Download Model"):
    with st.spinner("Downloading model from S3..."):
        download_dir(local_path, s3_prefix)
    st.success("Model Downloaded Successfully!")

text = st.text_area("Enter your review")

# --------------------------
# PREDICTION
# --------------------------
if st.button("Predict"):
    # Check if model folder exists
    if not os.path.isdir(local_path):
        st.error("Model not downloaded. Please click 'Download Model' first.")
    else:
        with st.spinner("Loading model and predicting..."):
            classifier = pipeline(
                "text-classification",
                model=local_path,
                device=0 if torch.cuda.is_available() else -1
            )

            output = classifier(text)
            st.write(output)
