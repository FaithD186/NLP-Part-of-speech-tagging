# HMM-part-of-speech-tagger

Created a Hidden Markov Model (HMM) for part-of-speech(POS) tagging, including training probability tables (i.e., initial, transition and emission) for HMM from training files containing text-tag pairs, and performing inference with the trained HMM to predict appropriate POS tags for untagged text.

The following command would train the program on the training data in training0.txt, testing on test0.txt, and storing the test output (POS predictions) in output0.txt:

```shell
python3 tagger.py -d training0.txt -t test0.txt -o output0.txt
```
