import json

def read_single_schema(path):
    with open(path) as file:
        return json.loads(file.readline())

def read_schemata(context_schema_path, sequence_schema_path):
    context_schema = read_single_schema(context_schema_path)
    sequence_schema = read_single_schema(sequence_schema_path)
    return context_schema, sequence_schema






if __name__ == '__main__':
    print('hallo')