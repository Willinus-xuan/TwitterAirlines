import pandas as pd
import re
import numpy as np
import nltk


df_gender = pd.read_csv('./cache/output/race_gender.csv',index_col=0,low_memory=False)
from transformers import pipeline
from datasets import load_dataset

from datasets import load_dataset_builder
ds_builder = load_dataset_builder("HuggingFaceM4/FairFace")

# Inspect dataset description
ds_builder.info.description

# Inspect dataset features
ds_builder.info.features

from datasets import load_dataset

dataset = load_dataset("HuggingFaceM4/FairFace", split="train")

dataset[2]['image']

from transformers import AutoFeatureExtractor
from datasets import load_dataset, Image

feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
dataset = load_dataset("beans", split="train")
from torchvision.transforms import RandomRotation

rotate = RandomRotation(degrees=(0, 90))
def transforms(examples):
    examples["pixel_values"] = [rotate(image.convert("RGB")) for image in examples["image"]]
    return examples

dataset.set_transform(transforms)
dataset[0]

