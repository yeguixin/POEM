import json
import os
import random
import sys
import time
from typing import Dict, Optional, Callable, Any

import numpy as np
import tensorflow as tf
from tensorflow.python.training.tracking import data_structures as tf_data_structures
from dpu_utils.utils import RichPath
from tensorflow_core.python.keras import Sequential
from tensorflow_core.python.keras.layers import Dense

from ..data import DataFold, GraphDataset
from ..layers import get_known_message_passing_classes
from ..models import GraphTaskModel
from .model_utils import save_model, load_weights_verbosely, get_model_and_dataset
from .task_utils import get_known_tasks
import nni

from memory_profiler import profile

def make_run_id(model_name: str, task_name: str, run_name: Optional[str] = None) -> str:
    """Choose a run ID, based on the --run-name parameter and the current time."""
    if run_name is not None:
        return run_name
    else:
        return "%s_%s__%s" % (model_name, task_name, time.strftime("%Y-%m-%d_%H-%M-%S"))


def log_line(log_file: str, msg: str):
    with open(log_file, "a") as log_fh:
        log_fh.write(msg + "\n")
    print(msg)


def train(
    model: GraphTaskModel,
    dataset: GraphDataset,
    dataset2:GraphDataset,
    dataset3:GraphDataset,
    log_fun: Callable[[str], None],
    run_id: str,
    max_epochs: int,
    patience: int,
    save_dir: str,
    quiet: bool = False,
    aml_run=None,
):
    train_data = dataset.get_tensorflow_dataset(DataFold.TRAIN).prefetch(3)
    valid_data = dataset.get_tensorflow_dataset(DataFold.VALIDATION).prefetch(3)
    #
    train_data_2 = dataset2.get_tensorflow_dataset(DataFold.TRAIN).prefetch(3)
    valid_data_2 = dataset2.get_tensorflow_dataset(DataFold.VALIDATION).prefetch(3)

    ##new_inputs
    train_data_3 = dataset3.get_tensorflow_dataset(DataFold.TRAIN).prefetch(3)
    valid_data_3 = dataset3.get_tensorflow_dataset(DataFold.VALIDATION).prefetch(3)


    save_file = os.path.join(save_dir, f"{run_id}_best.pkl")
##new_inputs
    _, _, initial_valid_results = model.run_one_epoch_new(valid_data,valid_data_2, valid_data_3 ,training=False, quiet=quiet)
    best_valid_metric, best_val_str = model.compute_epoch_metrics(initial_valid_results)
    log_fun(f"Initial valid metric: {best_val_str}.")
    save_model(save_file, model, dataset)
    best_valid_epoch = 0
    train_time_start = time.time()
    for epoch in range(1, max_epochs + 1):
        log_fun(f"== Epoch {epoch}")
        train_loss, train_speed, train_results = model.run_one_epoch_new(
            train_data,train_data_2, train_data_3,training=True, quiet=quiet
        )
        train_metric, train_metric_string = model.compute_epoch_metrics(train_results)
        log_fun(
            f" Train:  {train_loss:.4f} loss | {train_metric_string} | {train_speed:.2f} graphs/s",
        )
        valid_loss, valid_speed, valid_results = model.run_one_epoch_new(
            valid_data,valid_data_2, valid_data_3,training=False, quiet=quiet
        )
        valid_metric, valid_metric_string = model.compute_epoch_metrics(valid_results)
        nni.report_intermediate_result(float(valid_metric_string.split(" ")[-1]))
        log_fun(
            f" Valid:  {valid_loss:.4f} loss | {valid_metric_string} | {valid_speed:.2f} graphs/s",
        )

        if aml_run is not None:
            aml_run.log("task_train_metric", float(train_metric))
            aml_run.log("train_speed", float(train_speed))
            aml_run.log("task_valid_metric", float(valid_metric))
            aml_run.log("valid_speed", float(valid_speed))

        # Save if good enough.
        if valid_metric < best_valid_metric:
            log_fun(
                f"  (Best epoch so far, target metric decreased to {valid_metric:.5f} from {best_valid_metric:.5f}.)",
            )
            save_model(save_file, model, dataset)
            best_valid_metric = valid_metric
            best_valid_epoch = epoch
        elif epoch - best_valid_epoch >= patience:
            total_time = time.time() - train_time_start
            log_fun(
                f"Stopping training after {patience} epochs without "
                f"improvement on validation metric.",
            )
            log_fun(f"Training took {total_time}s. Best validation metric: {best_valid_metric}", )
            break
    return save_file


def unwrap_tf_tracked_data(data: Any) -> Any:
    if isinstance(data, (tf_data_structures.ListWrapper, list)):
        return [unwrap_tf_tracked_data(e) for e in data]
    elif isinstance(data, (tf_data_structures._DictWrapper, dict)):
        return {k: unwrap_tf_tracked_data(v) for k, v in data.items()}
    else:
        return data


#@profile(precision=4,stream=open('memory_profiler.log','w+'))
def run_train_from_args(args, hyperdrive_hyperparameter_overrides: Dict[str, str] = {}) -> None:
    # Get the housekeeping going and start logging:
    os.makedirs(args.save_dir, exist_ok=True)
    run_id = make_run_id(args.model, args.task)
    log_file = os.path.join(args.save_dir, f"{run_id}.log")
    
    def log(msg):
        log_line(log_file, msg)

    log(f"Setting random seed {args.random_seed}.")
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    tf.random.set_seed(args.random_seed)

    data_path = RichPath.create(args.data_path, args.azure_info)
    #second path
    data_path_2 = RichPath.create(os.path.split(args.data_path)[0]+'/test2', args.azure_info)

    data_path_3 = RichPath.create(os.path.split(args.data_path)[0] + '/test3', args.azure_info)

    ##new_inputs

    try:
        dataset, model = get_model_and_dataset(
            msg_passing_implementation=args.model,
            task_name=args.task,
            data_path=data_path,
            trained_model_file=args.load_saved_model,
            cli_data_hyperparameter_overrides=args.data_param_override,
            cli_model_hyperparameter_overrides=args.model_param_override,
            hyperdrive_hyperparameter_overrides=hyperdrive_hyperparameter_overrides,
            folds_to_load={DataFold.TRAIN, DataFold.VALIDATION},
            load_weights_only=args.load_weights_only,
            case_name = args.case, # add by zjq
        )
        #second
        dataset2, model_2 = get_model_and_dataset(
            msg_passing_implementation=args.model,
            task_name=args.task,
            data_path=data_path_2,
            trained_model_file=args.load_saved_model,
            cli_data_hyperparameter_overrides=args.data_param_override,
            cli_model_hyperparameter_overrides=args.model_param_override,
            hyperdrive_hyperparameter_overrides=hyperdrive_hyperparameter_overrides,
            folds_to_load={DataFold.TRAIN, DataFold.VALIDATION},
            load_weights_only=args.load_weights_only,
            case_name = args.case, # add by zjq
        )

        ##new_inputs
        dataset3, model_3 = get_model_and_dataset(
            msg_passing_implementation=args.model,
            task_name=args.task,
            data_path=data_path_3,
            trained_model_file=args.load_saved_model,
            cli_data_hyperparameter_overrides=args.data_param_override,
            cli_model_hyperparameter_overrides=args.model_param_override,
            hyperdrive_hyperparameter_overrides=hyperdrive_hyperparameter_overrides,
            folds_to_load={DataFold.TRAIN, DataFold.VALIDATION},
            load_weights_only=args.load_weights_only,
            case_name = args.case, # add by zjq
        )
    except ValueError as err:
        print(err.args)

    log(f"Dataset parameters: {json.dumps(unwrap_tf_tracked_data(dataset._params))}")
    log(f"Model parameters: {json.dumps(unwrap_tf_tracked_data(model._params))}")

    if args.azureml_logging:
        from azureml.core.run import Run

        aml_run = Run.get_context()
    else:
        aml_run = None
    if not args.load_trained_model:
        trained_model_path = train(
            model,
            dataset,
            dataset2,
            dataset3,
            log_fun=log,
            run_id=run_id,
            max_epochs=args.max_epochs,
            patience=args.patience,
            save_dir=args.save_dir,
            quiet=args.quiet,
            aml_run=aml_run,
        )
    else:
        trained_model_path = args.load_trained_model
    
    #new_inputs
    if args.run_test:
        data_path = RichPath.create(args.data_path, args.azure_info)
        data_path_2 = RichPath.create(os.path.split(args.data_path)[0] + '/test2', args.azure_info)
        data_path_3 = RichPath.create(os.path.split(args.data_path)[0] + '/test3', args.azure_info)
        log("== Running on test dataset")
        log(f"Loading data from {data_path}.")
        dataset.load_data(data_path, {DataFold.TEST})
        dataset2.load_data(data_path_2, {DataFold.TEST})
        dataset3.load_data(data_path_3, {DataFold.TEST})
        log(f"Restoring best model state from {trained_model_path}.")
        load_weights_verbosely(trained_model_path, model)
        test_data_1 = dataset.get_tensorflow_dataset(DataFold.TEST)
        test_data_2 = dataset2.get_tensorflow_dataset(DataFold.TEST)
        test_data_3 = dataset3.get_tensorflow_dataset(DataFold.TEST)
        _, _, test_results = model.run_one_epoch_new(test_data_1,test_data_2,test_data_3 ,training=False, quiet=args.quiet)
        test_metric, test_metric_string = model.compute_epoch_metrics(test_results)
        log(test_metric_string)
        nni.report_final_result(float(test_metric_string.split(" ")[-1]))
        if aml_run is not None:
            aml_run.log("task_test_metric", float(test_metric))


def get_train_cli_arg_parser():
    import argparse

    parser = argparse.ArgumentParser(description="Train a GNN model.")
    # We use a somewhat horrible trick to support both
    #  train.py --model MODEL --task TASK --data_path DATA_PATH
    # as well as
    #  train.py model task data_path
    # The former is useful because of limitations in AzureML; the latter is nicer to type.
    

    # add by zjq
    if "--model" in sys.argv:
        case_param_name, model_param_name, task_param_name, data_path_param_name = "--case","--model", "--task", "--data_path"
    else:
        case_param_name, model_param_name, task_param_name, data_path_param_name = "case","model", "task", "data_path"

    # add by zjq
    parser.add_argument(
        case_param_name,
        type=str,
        choices=sorted(["case1","case2","case3","case4"]),
        help="case type to train.",
    )
    '''
    parser.add_argument(
        "run_mode",
        type=int,
        choices=sorted([1,10]),
        help="case run mode type to train.",
    )'''


    parser.add_argument(
        model_param_name,
        type=str,
        choices=sorted(get_known_message_passing_classes()),
        help="GNN model type to train.",
    )
    parser.add_argument(
        task_param_name,
        type=str,
        choices=sorted(get_known_tasks()),
        help="Task to train model for.",
    )
    parser.add_argument(
        data_path_param_name, 
        type=str, 
        help="Directory containing the task data."
    )
    
    parser.add_argument(
        "--save-dir",
        dest="save_dir",
        type=str,
        default="trained_model_case1",
        help="Path in which to store the trained model and log.",
    )
    parser.add_argument(
        "--load-trained-model",
        dest="load_trained_model",
        type=str,
        help="model train load zjq",
    )
    parser.add_argument(
        "--model-params-override",
        dest="model_param_override",
        type=str,
        help="JSON dictionary overriding model hyperparameter values.",
    )
    parser.add_argument(
        "--data-params-override",
        dest="data_param_override",
        type=str,
        help="JSON dictionary overriding data hyperparameter values.",
    )
    parser.add_argument(
        "--max-epochs",
        dest="max_epochs",
        type=int,
        default=10000,
        help="Maximal number of epochs to train for.",
    )
    parser.add_argument(
        "--patience",
        dest="patience",
        type=int,
        default=25,
        help="Maximal number of epochs to continue training without improvement.",
    )
    parser.add_argument(
        "--seed", dest="random_seed", type=int, default=0, help="Random seed to use.",
    )
    parser.add_argument(
        "--run-name", dest="run_name", type=str, help="A human-readable name for this run.",
    )
    parser.add_argument(
        "--azure-info",
        dest="azure_info",
        type=str,
        default="azure_auth.json",
        help="Azure authentication information file (JSON).",
    )
    parser.add_argument(
        "--load-saved-model",
        dest="load_saved_model",
        help="Optional location to load initial model weights from. Should be model stored in earlier run.",
    )
    parser.add_argument(
        "--load-weights-only",
        dest="load_weights_only",
        action="store_true",
        help="Optional to only load the weights of the model rather than class and dataset for further training (used in fine-tuning on pretrained network). Should be model stored in earlier run.",
    )
    parser.add_argument(
        "--quiet", dest="quiet", action="store_true", help="Generate less output during training.",
    )
    parser.add_argument(
        "--run-test",
        dest="run_test",
        action="store_true",
        default=True,
        help="Run on testset after training.",
    )
    parser.add_argument(
        "--azureml_logging",
        dest="azureml_logging",
        action="store_true",
        help="Log task results using AML run context.",
    )
    parser.add_argument("--debug", dest="debug", action="store_true", help="Enable debug routines")

    parser.add_argument(
        "--hyperdrive-arg-parse",
        dest="hyperdrive_arg_parse",
        action="store_true",
        help='Enable hyperdrive argument parsing, in which unknown options "--key val" are interpreted as hyperparameter "key" with value "val".',
    )

    return parser
