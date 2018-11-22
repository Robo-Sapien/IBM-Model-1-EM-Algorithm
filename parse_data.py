import json

def load_data_from_json(filename):
    '''
    This function will read the json file and give out the data
    for furthur processing.
    '''
    with open(filename) as fhandle:
        data=json.load(fhandle)
        #print data
    return data


if __name__=='__main__':
    filename='corpus/data1.json'

    load_data_from_json(filename)
