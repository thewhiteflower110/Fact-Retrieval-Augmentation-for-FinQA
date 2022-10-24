pwd Fact-Retrieval-Augmentation-for-FinQA/
python setup.py install
python3 train.py --option pretrain --epochs 10 --lr 1e-3 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --dev_out "cfimdb-dev-output.txt" --test_out "cfimdb-test-output.txt" --use_gpu
