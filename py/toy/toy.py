import json


def parse_samples(fn):
    # read file fn, which is a jsonl file.
    # return a list of lists, where each list is a sample
    # and each sample is a list of tokens
    with open(fn, 'r') as f:
        samples = f.readlines()
        ret = []
        for s in samples:
            s = json.loads(s)
            print(s)
            ret.append(s)
    return ret


def write_samples(samples, fn):
    # write samples to file fn, which is a jsonl file
    with open(fn, 'w') as f:
        for s in samples:
            line = '''{"input": [{"role": "system", "content": "You are a helpful assistant."}, ''' + \
                '''{"role": "user", "content": "''' + \
                s['problem'] + '''"}], "ideal": '''

            # if s['answer'] is an integer, print it
            # otherwise, print it as a string
            if isinstance(s['answer'], int):
                line += str(s['answer'])
            else:
                line += '''"''' + s['answer'] + '''"'''
            line += '''}\n'''
            f.write(line)


if __name__ == '__main__':
    ss = parse_samples("/tmp/samples.jsonl")
    write_samples(ss, "/tmp/out.jsonl")
