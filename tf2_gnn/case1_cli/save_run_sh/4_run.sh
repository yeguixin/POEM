rm -rf trained* emb* *log
cp ../cli_utils/training_utils_case4.py ../cli_utils/training_utils.py
cp ../data/graph_dataset_case4.py ../data/graph_dataset.py
cp ../data/jsonl_graph_property_dataset_case4.py ../data/jsonl_graph_property_dataset.py
cp ../models/graph_binary_classification_task_case4.py ../models/graph_binary_classification_task.py
cp ../models/graph_task_model_case4.py ../models/graph_task_model.py
CUDA_VISIBLE_DEVICES=0 python3 -u train.py case4 RGIN GraphBinaryClassification /dev/shm/zjq/data/CWE-200
