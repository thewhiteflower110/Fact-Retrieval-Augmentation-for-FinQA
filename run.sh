pwd Fact-Retrieval-Augmentation-for-FinQA/
python3 setup.py install
python3 train_fr_module.py --option pretrain --epochs 10 --lr 1e-3 --train data/cfimdb-train.txt --dev data/cfimdb-dev.txt --test data/cfimdb-test.txt --dev_out "cfimdb-dev-output.txt" --test_out "cfimdb-test-output.txt" --use_gpu
python3 convert_fr_outputs.py
