import json

file_path = ['train.jsonl', 'dev.jsonl']


for file in file_path:
    dataset = []
    with open(file, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            data['context'] = data['passage'] + ' ' + data['question'] + '?'
            data['completion'] = 'true' if data['answer'] else 'false'
            del data['question']
            del data['title']
            del data['answer']
            del data['passage']

            dataset.append(data)

    with open(file, 'w', encoding='utf-8') as f:
        for data in dataset:
            f.write(f"{json.dumps(data, ensure_ascii=False)}\n")