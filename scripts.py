import os
import math
import pickle
import pandas as pd


def read_saves(save_path: str):
    if not os.path.exists(save_path):
        return dict()
    with open(save_path, "rb") as stream:
        saves = pickle.load(stream)
    return saves


def write_saves(save_path: str, saves: dict):
    with open(save_path, "wb") as stream:
        pickle.dump(saves, stream)


def record_saves(save_path: str, save_nb: int, history, model):
    saves: dict = read_saves(save_path)
    if len(saves) != 0:
        max_val_loss = max(saves, key=saves.get)
    else:
        max_val_loss = math.inf
    history = pd.DataFrame(history.history)
    history_min_val_loss = min(history["val_loss"].values.tolist())
    if history_min_val_loss <= max_val_loss:
        saves[history_min_val_loss] = (model.to_json(), history)
        saves = dict(sorted(saves.items()))
        if len(saves) > save_nb:
            max_val_loss = max(saves, key=saves.get)
            del saves[max_val_loss]
        write_saves(save_path, saves)
    return saves
