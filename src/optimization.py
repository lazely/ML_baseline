import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src import train
import optuna
from optuna.samplers import TPESampler

from src.utils.params import get_params

def get_objective(bconfig):
    def objective(trial):
        config = get_params(bconfig, trial=trial)
        return train.run(config)
    return objective

def run(config):
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    objective = get_objective(config)
    study.optimize(objective, n_trials=config['hyperparameter_optimization']['n_trials'])
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("   {}: {}".format(key, value))
    return trial.params