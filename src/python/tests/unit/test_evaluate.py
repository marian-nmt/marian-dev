from pymarian import Evaluator
from pymarian.utils import get_known_model


def test_evaluator_chrfoid():
    model_path, vocab_path = get_known_model("chrfoid-wmt23")
    args = dict(
        like="comet-qe",
        model=model_path,
        mini_batch=8,
        maxi_batch=64,
        cpu_threads=4,
        workspace=8000,
        quiet='',
        vocabs=[vocab_path, vocab_path],
    )
    #args = dict(help='')   # to get help message with all args
    eval = Evaluator(**args)
    data = [
        ["This is a test", "This is a test A"],
        ["This is a test B", "This is a test C"],
        ["This is a test D", "This is a test E"],
    ]
    expected_scores = [
        0.0548, 
        0.0797,
        0.0988
    ]
    
    scores = eval.run(data)
    assert len(scores) == len(data)
    for score, expected_score in zip(scores, expected_scores):
        if isinstance(score, list):
            score = score[0]
        assert abs(score - expected_score) < 0.0001
        
    