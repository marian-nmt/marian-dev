import numpy as np
import cupy as cp
import sys
import argparse
from pathlib import Path

parser = argparse.ArgumentParser(description='Apply MBR with COMET top layers')
parser.add_argument('-m', '--model',  type=Path,  help='COMET model path', required=True)
parser.add_argument('-s', '--source', type=Path,  help='Source file embeddings', required=True)
parser.add_argument('-t', '--hyps',  type=Path,   help='Sample file embeddings', required=True)
parser.add_argument('--num_source', type=int, help='Number of sentence', required=True)
parser.add_argument('--num_hyps', type=int, help='Number of samples per sentence', required=True)
parser.add_argument('--fp16', help='Use fp16 for computation', action='store_true')
parser.add_argument('--batch_size', type=int, help='Batch size during MBR', default=32)
parser.add_argument('-d', '--devices', nargs='+', type=int, help="GPU device id to use", default=[0, 1, 2, 3, 4, 5, 6, 7])
args = parser.parse_args()


model_path   = args.model
src_emb_path = args.source 
smp_emb_path = args.hyps

num_sents = args.num_source
num_samps = args.num_hyps

emb_size = 1024

compute_type=cp.float32
if args.fp16:
    compute_type=cp.float16

batch_size = args.batch_size
devices = args.devices

sources = np.memmap(src_emb_path, mode='r', dtype=np.float32, shape=(num_sents, emb_size))
samples = np.memmap(smp_emb_path, mode='r', dtype=np.float32, shape=(num_sents, num_samps, emb_size))

def mbr_decode_batch(pooler, mt, src, ref):
    batch_size = mt.shape[0]

    diffRef = abs(mt - ref)
    prodRef = mt * ref

    diffSrc = cp.repeat(abs(mt - src), repeats=num_samps, axis=-2);
    prodSrc = cp.repeat(mt * src,      repeats=num_samps, axis=-2);

    mt  = cp.repeat(mt,  repeats=num_samps, axis=-2)
    ref = cp.repeat(ref, repeats=batch_size,  axis=-3) 

    emb = cp.concatenate([mt, ref, prodRef, diffRef, prodSrc, diffSrc], axis=-1)
    
    layer1    = cp.tanh(cp.dot(emb,    pooler[0]["weight"]) + pooler[0]["bias"])
    layer2    = cp.tanh(cp.dot(layer1, pooler[1]["weight"]) + pooler[1]["bias"]) 
    comet     =         cp.dot(layer2, pooler[2]["weight"]) + pooler[2]["bias"]
    
    mbr_score = cp.reshape(cp.mean(comet, axis=-2), (batch_size,))
    
    return mbr_score

    
def mbr_decode(pooler, i, batch_size=50):
    sources_gpu = cp.asarray(sources[i, :], compute_type)
    samples_gpu = cp.asarray(samples[i, :, :], compute_type)

    src = cp.reshape(sources_gpu, (1, 1, emb_size))
    mt  = cp.reshape(samples_gpu, (num_samps, 1, emb_size))
    ref = cp.reshape(mt, (1, num_samps, emb_size))

    batches = cp.array_split(mt, int(num_samps / batch_size))
    scores = []
    for batch in batches:
        mbr_scores_batch = mbr_decode_batch(pooler, batch, src, ref)
        scores.append(mbr_scores_batch)
    
    mbr_scores = cp.concatenate(scores, axis=-1)
    best_index = cp.argmax(mbr_scores, axis=-1)
    best_score = cp.max(mbr_scores, axis=-1)
    
    return best_index, best_score

def consume(k):
    j = 0
    candidates = []
    for line in sys.stdin:
        line = line.rstrip()
        candidates.append(line)

        if len(candidates) == num_samps:
            best = best_gpu[k + j]
            best_index = cp.asnumpy(best[0])
            best_score = cp.asnumpy(best[1]) 
            print(f"{k + j}\t{best_index}\t{best_score:.4f}\t{candidates[best_index]}")
            candidates = []
            j += 1
            if j == step:
                k += step
                break
    return k

#####################################################

model = np.load(model_path)

poolers = []
for id in devices:
    with cp.cuda.Device(id):
        pooler = []
        for i, layerNo in enumerate([0, 3, 6]):
            w = cp.asarray(model[f"CometQEPooler->layers->at({layerNo})->as<marian::nn::Linear>()->weight"], compute_type)
            b = cp.asarray(model[f"CometQEPooler->layers->at({layerNo})->as<marian::nn::Linear>()->bias"],   compute_type)
            pooler.append({"weight": w, "bias": b})
        poolers.append(pooler)

step = batch_size
best_gpu = []
k = 0
for i in range(num_sents):
    gpu_id = i % len(devices)
    with cp.cuda.Device(devices[gpu_id]):
        best_gpu.append(mbr_decode(poolers[gpu_id], i, batch_size=batch_size))    
    if len(best_gpu) % step == 0:
        k = consume(k)

# get rest
k = consume(k)
