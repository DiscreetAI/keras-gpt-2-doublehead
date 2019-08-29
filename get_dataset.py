import json
from pytorch_pretrained_bert import cached_path
from pytorch_pretrained_bert import OpenAIGPTTokenizer

tokenizer = OpenAIGPTTokenizer.from_pretrained('openai-gpt')
url = "s3://datasets.huggingface.co/personachat/personachat_self_original.json"

# Download and load JSON dataset
personachat_file = cached_path(url)
with open(personachat_file, "r", encoding="utf-8") as f:
    dataset = json.loads(f.read())

# with open('dataset.json', "w", encoding="utf-8") as f:
#     f.write(json.dumps(dataset))

# Tokenize and encode the dataset using our loaded GPT tokenizer
def tokenize(obj):
    if isinstance(obj, str):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
    if isinstance(obj, dict):
        return dict((n, tokenize(o)) for n, o in obj.items())
    return list(tokenize(o) for o in obj)
 
dataset = tokenize(dataset)