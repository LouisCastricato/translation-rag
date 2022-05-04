import numpy 
import torch
import json

# load a csv that is space delimited
def load_space_delimited(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip().split(' ') for line in lines]

# save a dictionary to a csv
def save_dict(d, file_path):
    with open(file_path, 'w') as f:
        for k, v in d.items():
            f.write(k + ' ' + v + '\n')
            
#converts a list of lists to a dictionary
def list_to_dict(data):
    return {l[0]: l[1] for l in data}

# saves a dictionary of torch tensors to a json
def save_to_json(d, file_path):
    json_dump = json.dumps(d)
    with open(file_path, 'w') as f:
        f.write(json_dump)

# stacks torch dictionaries along the 0th dimension per key
def stack_dicts(list_of_dicts):
    stacked = {}
    for d in list_of_dicts:
        for k, v in d.items():
            if k not in stacked:
                stacked[k] = v
            else:
                stacked[k] = torch.cat((stacked[k], v), 0)
    return stacked

# converts a dictionary of pytorch tensors to lists of lists
def convert_dict_of_tensors_to_list(d):
    return {k: v.numpy().tolist() for k, v in d.items()}

def load_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# test the above function
if __name__ == '__main__':
    d1 = {'a': torch.zeros((1,3)), 'b': torch.ones((1,3)), 'c': torch.zeros((1,3))}
    d2 = {'a': torch.zeros((1,3)), 'b': torch.ones((1,3)), 'c': torch.zeros((1,3))}
    d3 = convert_dict_of_tensors_to_list(stack_dicts([d1, d2]))

    save_to_json(d3, 'test.json')