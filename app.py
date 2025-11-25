import streamlit as st
import os
import torch
from transformers import pipeline
import boto3

bucket_name = "sentiment-analysis-tinybert-sujan"
local_path = 'tinybert-sentiment-analysis'
s3_prefix = 'ml-models/tinybert-sentiment-analysis/'

s3 = boto3.client('s3')

def download_dir(local_path, s3_prefix):
    os.makedirs(local_path, exist_ok=True)
    paginator = s3.get_paginator('list_objects_v2')
    for result in paginator.paginate(Bucket=bucket_name, Prefix=s3_prefix):
        if 'Contents' in result:
            for key in result['Contents']:
                s3_key = key['Key']
                local_file = os.path.join(local_path, os.path.relpath(s3_key, s3_prefix))
                os.makedirs(os.path.dirname(local_file), exist_ok=True)
                s3.download_file(bucket_name, s3_key, local_file)

st.title("Sentiment Analysis using TinyBERT Model")

if st.button("Download Model"):
    with st.spinner("Downloading Model..."):
        download_dir(local_path, s3_prefix)
        st.success("Model downloaded successfully!")

text = st.text_area("Enter Your Review")

if st.button("Predict"):
    with st.spinner("Loading model & predicting..."):

        # Load from LOCAL FOLDER, not model name
        classifier = pipeline(
            "text-classification",
            model=local_path,     # <--- FIX
            device=0 if torch.cuda.is_available() else -1
        )
        output = classifier(text)
        st.write(output)
