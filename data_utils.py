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
