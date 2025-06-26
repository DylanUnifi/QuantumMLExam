import yaml

def load_config(path="/data01/pc24dylfou/PycharmProjects/qml_Project/configs/train_classical.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)
