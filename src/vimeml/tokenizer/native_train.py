"""Child process capturing SentencePiece's native C++ training diagnostics."""
import json
import sys
from pathlib import Path


def main():
    import sentencepiece as spm
    request = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    spm.set_random_generator_seed(request['seed'])
    spm.SentencePieceTrainer.train(**request['options'])


if __name__=='__main__': main()
