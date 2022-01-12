import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel

from transformers import logging
logging.set_verbosity_error()

def predict():
    meta_data = joblib.load("meta.bin")
    enc_tag = meta_data["enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    sentence = """
    O presidente Bolsonaro questionou a ANVISA (Agência Nacional de Vigilância Sanitária) sobre a vacinação infantil durante o discurso em São Paulo, conforme reportou a Folha de São Paulo.
    """
    sentence = sentence.split()

    tokenized_sentence = config.TOKENIZER.encode(sentence)
    print(tokenized_sentence)
    
    test_dataset = dataset.EntityDataset(
        texts=[sentence],
        tags=[[0] * len(sentence)]
    )

    device = torch.device("cuda")

    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.MODEL_PATH))
    model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)

        labels = enc_tag.inverse_transform(
                    tag.argmax(2).cpu().numpy().reshape(-1)
                )[1:len(tokenized_sentence)-1]

    for i in range(len(sentence)):
        print(f"{sentence[i]}\t{labels[i]}")

predict()
