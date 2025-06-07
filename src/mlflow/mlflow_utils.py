import mlflow

def init_mlflow_run(config):
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])
    run = mlflow.start_run(run_name=config["mlflow"]["run_name"])

    # Helper function to flatten nested config dict
    def flatten_dict(d, parent_key='', sep='.'):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_params = flatten_dict(config)
    mlflow.log_params(flat_params)

    return run
