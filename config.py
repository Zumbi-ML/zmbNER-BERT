from transformers import BertTokenizer
from transformers import AutoModel, AutoTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10

#BASE_MODEL_PATH = "bert-base-uncased"
BASE_MODEL_PATH = "neuralmind/bert-base-portuguese-cased"

MODELS_PATH = "models/"
PRETRAINED = ""
VERSION = "v1.0"

MODEL_PATH = "model.bin"
TRAINING_FILE = "input/zmb-iob-annotations-train-full.tsv"

#TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
#TOKENIZER = AutoTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
TOKENIZER = BertTokenizer.from_pretrained('neuralmind/bert-base-portuguese-cased')
