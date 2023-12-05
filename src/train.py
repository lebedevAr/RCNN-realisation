import yaml
from functions import train_model


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["train"]

    epochs_num = params['epochs_num']
    dataset_path = params["dataset_path"]
    weights_name_outs = params["weights_name_outs"]

    train_model(dataset_path, epochs_num, weights_name_outs)
