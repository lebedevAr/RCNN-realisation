import yaml
from functions import evaluate_model


if __name__ == "__main__":
    params = yaml.safe_load(open("params.yaml"))["evaluate"]

    weights_name_in = params["weights_name_in"]
    dataset_path = params["dataset_path"]
    detection_threshold = params['detection_threshold']


    print(evaluate_model(weights_name_in, dataset_path, detection_threshold))