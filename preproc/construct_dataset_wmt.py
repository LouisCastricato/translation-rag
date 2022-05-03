from transformers import FSMTForConditionalGeneration, FSMTTokenizer
import pandas as pd
from tqdm import tqdm
import torch
from torch.nn.utils.rnn import pad_sequence
# load ../unigram_freq.csv to a list of lists
data = pd.read_csv('../unigram_freq.csv', header=None).values.tolist()
# keep only the first column of data
data = [row[0] for row in data]
#print(data[:10])
# remove all words with less than 4 characters
data = list(filter(lambda x: len(str(x)) > 4, data))

start = 200
slice_amt = 100000
data = data[start:start+slice_amt]

mname = "facebook/wmt19-en-ru"
tokenizer = FSMTTokenizer.from_pretrained(mname)
model = FSMTForConditionalGeneration.from_pretrained(mname).cuda()
model.eval()

# this helper function saves a list of tuples to a csv
def save_to_csv(data, filename):
    with open(filename, 'w') as f:
        for row in data:
            f.write(str(row[0]) + ',' + str(row[1]) + '\n')
    
def translate(input_string):
    """"
    Translate a batch of words using the model
    :param input: a batch of words to translate
    """
    input_string = list(map(lambda x: str(x), input_string))
    # weird bug with tokenizer but whatever
    toks = list(map(lambda x: tokenizer.encode(x), input_string))
    # convert toks to a list of tensors
    toks = list(map(lambda x: torch.tensor(x).cuda(), toks))
    # pad the tensors
    toks = pad_sequence(toks, batch_first=True, padding_value=tokenizer.pad_token_id)

    outputs = model.generate(toks)
    decoded = list(map(lambda x: tokenizer.decode(x, skip_special_tokens=True), outputs))
    return decoded

bs = 100
translation_list = list()
list_of_tuples = list()
# batch over data to translate
for i in tqdm(range(0, len(data), bs)):
    words = data[i:i+bs]
    translated_words = translate(words)
    translation_list += translated_words
    list_of_tuples += list(zip(words, translated_words))


# save to file
save_to_csv(list_of_tuples, '../english_russian_dataset.csv')

