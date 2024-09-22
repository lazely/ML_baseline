from src import train
import optuna
from optuna.samplers import TPESampler

from src.utils.params import get_params

def objective(trial):
    config = get_params(trial=trial)
    
    return train.run(config)

def run(config):
    hp_config = config['hyperparameter_optimization']
    study = optuna.create_study(direction="maximize", sampler=TPESampler())
    study.optimize(objective, n_trials=hp_config['n_trials'])
    print("Best trial:")
    trial = study.best_trial
    print("  Value: ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print("   {}: {}".format(key, value))
    return trial.params