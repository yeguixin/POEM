rm -rf trained* emb* *log
cp ../cli_utils/training_utils_case3.py ../cli_utils/training_utils.py
cp ../data/graph_dataset_case3.py ../data/graph_dataset.py
cp ../data/jsonl_graph_property_dataset_case3.py ../data/jsonl_graph_property_dataset.py
cp ../models/graph_binary_classification_task_case3.py ../models/graph_binary_classification_task.py
cp ../models/graph_task_model_case3.py ../models/graph_task_model.py
python -u train.py case3 RGIN GraphBinaryClassification /dev/shm/zjq/case3_data/ast/
