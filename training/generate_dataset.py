from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import pandas as pd
from tqdm import tqdm
import torch

# load ../unigram_freq.csv to a list of lists
data = pd.read_csv('../unigram_freq.csv', header=None).values.tolist()
# keep only the first column of data
data = [row[0] for row in data]
print(data[:10])
# remove all words with less than 4 characters
data = list(filter(lambda x: len(str(x)) > 4, data))


data = data[0:100]

mname = "facebook/wmt19-en-de"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname).cuda()

def translate(input):
    """"
    Translate a batch of words using the model
    :param input: a batch of words to translate
    """
    input = list(map(lambda x: str(x), input))
    input_ids = tokenizer.encode(input, return_tensors="pt").cuda()
    input_ids = torch.transpose(input_ids, 0, 1)
    print(tokenizer.decode(input_ids.tolist()[0]))
    outputs = model.generate(input_ids)

    decoded = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), outputs))
    return decoded
bs = 10
# batch over data to translate
for i in tqdm(range(0, len(data), bs)):
    words = data[i:i+bs]
    translation = translate(words)
    print(translation)

