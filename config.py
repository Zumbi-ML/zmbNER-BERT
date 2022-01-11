from transformers import BertTokenizer

MAX_LEN = 128
TRAIN_BATCH_SIZE = 32
VALID_BATCH_SIZE = 8
EPOCHS = 10
BASE_MODEL_PATH = "bert-base-uncased"
MODEL_PATH = "model.bin"
TRAINING_FILE = "input/ner_dataset_100.csv"
TOKENIZER = BertTokenizer.from_pretrained('bert-base-uncased')
