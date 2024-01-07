'''
decode predict jsonl file and compare it with reference answer (true or false)
'''
import json
import argparse
import encoder

parser = argparse.ArgumentParser(description='evaluate the performance on a model on boolq based on its output token id\'s')
parser.add_argument('--input', type=str, default='predict.jsonl', help='input file')
parser.add_argument('--ref', type=str, default='test.jsonl', help='ref file')
parser.add_argument('--output', type=str, default=None, help='output file')
parser.add_argument('--vocab', type=str, default=None, help='vocab path')


def compare(pred, ref) -> bool:
    return pred.lower().strip().strip('.').strip() == ref.lower().strip().strip('.').strip()

if __name__ == '__main__':
    args = parser.parse_args()
    outputs = {}
    with open(args.input, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            outputs[sample['id']] = sample['predict']

    # convert the tokens from the predictions in the beam decoding output into string
    encoder = encoder.get_encoder(args.vocab)
    output_text = []
    for i in range(len(outputs)):
        output_text.append(encoder.decode(outputs[i]).split('<|endoftext|>')[0].split('\n\n')[0].strip())

    if args.output is not None:
        with open(args.output, 'w') as f:
            for i in range(len(output_text)):
                f.write(output_text[i] + '\n')


    ref_answers = []
    with open(args.ref, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            ref_answers.append(sample['completion'])

    # compare the string whether it is true or false (yes or no)
    count = 0
    correct = 0
    for i in range(len(output_text)):
        assert i < len(ref_answers), 'output has more lines that ref'
        count += 1
        if compare(output_text[i], ref_answers[i]):
            correct += 1

    print(f'Correct: {correct}')
    print(f'Total: {count}')
    print(f'Accuracy: {correct/count}')