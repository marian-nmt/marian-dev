
import sys
import itertools
import argparse
import logging as log

from pymarian import Evaluator

DEF_BATCH_SIZE = 2
log.basicConfig(level=log.INFO, format='%(asctime)s %(levelname)s %(message)s')


def make_batches(lines, batch_size=DEF_BATCH_SIZE):
    iter = (line.spplit('\t') for line in lines)
    while True:
        chunk = list(itertools.islice(iter, batch_size))
        if not chunk:
            return
        yield chunk

def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", metavar='FILE', type=str, required=True,
                        help="model path")
    parser.add_argument("-v", "--vocabs", metavar='FILE', type=str, nargs="+", required=True, 
                        help="vocabs path")
    parser.add_argument("-b", "--batch-size", metavar='INT', type=int, default=DEF_BATCH_SIZE)
    parser.add_argument("-l", "--like", type=str, default="comet-qe",
                        help="model like: comet-qe, comet, bleurt, etc..")
    
    parser.add_argument("-d", "--devices", type=int, nargs="+", default=[0])
    parser.add_argument("-w", "--workspace", type=int, default=8*1024, 
                        help="Workspace memory in bytes. When this is negative number (for GPU devices), \
                            this is interpreted as as (total-workspace)",)
    return vars(parser.parse_args())

def main():
    args = parse_args()
    batch_size = args.pop("batch_size")
    args['vocabs'] = ' '.join(args['vocabs'])
    log.info(f'args: {args}')
    # TODO: support biding of dict or kwargs instead of CLI string
    cli_string = ""
    for name, val in args.items():
        name = name.replace('_', '-')
        if isinstance(val, list):
            val = ' '.join(map(str, val))
        cli_string += f"--{name} {val} "
    cli_string = cli_string.strip()
    #cli_string += " --quiet"
    #cli_string += " --log-level debug"
    evaluator = Evaluator(cli_string)
    log.info(f"====Loaded evaluator=========")
    batches = make_batches(sys.stdin, batch_size=batch_size)
    for batch in batches:
        scores = evaluator.run(batch)
        for ex, score in zip(batch, scores):
            print(f'{ex}\t{score[0]:.6f}')

if __name__ == "__main__":
   main() 