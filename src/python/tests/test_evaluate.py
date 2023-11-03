"""
# silense marian log
export MARIAN_QUIET=yes

# run all tests in this file
  pytest -v src/python/tests/test_evaluate.py 
  pytest -vx src/python/tests/test_evaluate.py   #stop on first failure

# run a single test:
   pytest -v src/python/tests/test_evaluate.py -k test_evaluator_chrfoid
   pytest -vs src/python/tests/test_evaluate.py -k test_evaluator_chrfoid # see stdout and stderr
"""
import os
from pymarian import Evaluator
from pymarian.utils import get_known_model
from itertools import zip_longest

QUIET = os.getenv('MARIAN_QUIET', "").lower() in ("1", "yes", "y", "true", "on")
CPU_THREADS = int(os.getenv('MARIAN_CPU_THREADS', "4"))
WORKSPACE_MEMORY = int(os.getenv('MARIAN_WORKSPACE_MEMORY', "6000"))

EPSILON = 0.0001   # the precision error we afford in float comparison 

BASE_ARGS = dict(
    mini_batch=8,
    maxi_batch=64,
    cpu_threads=CPU_THREADS,
    workspace=WORKSPACE_MEMORY,
    quiet=QUIET,
)

# dummy sentences for testing
SAMPLE_SRC_HYP = [
        ["This is a test", "This is a test A"],
        ["This is a test B", "This is a test C"],
        ["This is a test D", "This is a test E"],
    ]
SAMPLE_REF_HYP = SAMPLE_SRC_HYP   # same for now
SAMPLE_SRC_HYP_REF =  [
        ["This is a test", "This is a test A", "This is a test AA"],
        ["This is a test B", "This is a test C", "This is a test CC"],
        ["This is a test D", "This is a test E", "This is a test EE"],
    ]


def test_evaluator_chrfoid():
    model_path, vocab_path = get_known_model("chrfoid-wmt23")
    args = BASE_ARGS | dict(
        like="comet-qe",
        model=model_path,
        vocabs=[vocab_path, vocab_path],
    )
    #args = dict(help='')   # to get help message with all args
    eval = Evaluator(**args)
    data = SAMPLE_SRC_HYP
    expected_scores = [
        0.0548,
        0.0797,
        0.0988
    ]
 
    scores = eval.evaluate(data)
    assert len(scores) == len(data)
    for score, expected_score in zip(scores, expected_scores):
        if isinstance(score, list):
            score = score[0]
        assert abs(score - expected_score) < EPSILON
   



def test_evaluator_cometoid22_wmt22():
    model_path, vocab_path = get_known_model("cometoid22-wmt22")
    args = BASE_ARGS | dict(
        like="comet-qe",
        model=model_path,
        vocabs=[vocab_path, vocab_path],
    )
    #args = dict(help='')   # to get help message with all args
    eval = Evaluator(**args)
    data = SAMPLE_SRC_HYP
    expected_scores = [
        0.71845,
        0.7906, 
        0.81549
    ]
 
    scores = eval.evaluate(data)
    assert len(scores) == len(data)

    for score, expected_score in zip(scores, expected_scores):
        if isinstance(score, list):
            score = score[0]
        assert abs(score - expected_score) < EPSILON
        

def test_evaluator_cometoid22_wmt23():
    model_path, vocab_path = get_known_model("cometoid22-wmt23")
    args = BASE_ARGS | dict(
        like="comet-qe",
        model=model_path,
        vocabs=[vocab_path, vocab_path],
    )
    eval = Evaluator(**args)
    data = SAMPLE_SRC_HYP
    expected_scores = [0.75715, 0.81395, 0.8361]
 
    scores = eval.evaluate(data)
    assert len(scores) == len(data)
    for score, expected_score in zip(scores, expected_scores):
        if isinstance(score, list):
            score = score[0]
        assert abs(score - expected_score) < EPSILON
   

def test_evaluator_bleurt():
    model_path, vocab_path = get_known_model("bleurt20")
    args = BASE_ARGS | dict(
        like="bleurt",
        model=model_path,
        vocabs=[vocab_path, vocab_path],
    )
    
    eval = Evaluator(**args)
    data = SAMPLE_REF_HYP
    scores  = eval.evaluate(data)
    expected_scores = [0.30929, 0.3027, 0.3113]
    assert len(scores) == len(data)
    for score, expected_score in zip(scores, expected_scores):
        if isinstance(score, list):
            score = score[0]
        assert abs(score - expected_score) < EPSILON

# TODO: These below tests are failing

def test_evaluator_comet20qe():
    
    model_path, vocab_path = get_known_model("comet20-da-qe")
    args = BASE_ARGS | dict(
        like="comet-qe",
        model=model_path,
        vocabs=[vocab_path, vocab_path],
    )
    
    eval = Evaluator(**args)
    data = SAMPLE_SRC_HYP
    scores  = eval.evaluate(data)
    assert len(scores) == len(data)
    # TODO: add expected scores and asserts


def test_evaluator_comet20ref():    
    model_path, vocab_path = get_known_model("comet20-da")
    args = BASE_ARGS | dict(
        like="comet",
        model=model_path,
        vocabs=[vocab_path, vocab_path],
    )

    eval = Evaluator(**args)
    data = SAMPLE_SRC_HYP_REF
    scores  = eval.evaluate(data)
    len(scores) == len(data)
   # TODO: add expected scores and asserts
