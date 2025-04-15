import optuna
import numpy as np
import functions as fn
import utils as ut

def fixed_step_learner(trial):
    step = trial.suggest_float("step", 1e-12, 1, log=True)
    return ut.fixed_step(step)

def polynomial_decay_learner(trial):
    alpha = trial.suggest_float("alpha", 1e-12, 1, log=True)
    beta = trial.suggest_float("beta", 1e-12, 1, log=True)
    return ut.polynomial_decay(alpha, beta)

def wolfe_strategy_learner(trial):
    c1 = trial.suggest_float("c1", 1e-12, 1, log=True)
    c2 = trial.suggest_float("c2", c1, 1, log=True)
    wolfe_max_iter = trial.suggest_int("wolfe_max_iter", 10, 50)
    return lambda k, fu, ma: ut.wolfe(phi=fu, c1=c1, c2=c2, amax=ma, max_iter=wolfe_max_iter)

def method_learner(trial, f, sl, met, st, bnd):
    strategy = sl(trial)
    tol = trial.suggest_float("tol", 1e-12, 1e-3, log=True)
    desc_max_iter = trial.suggest_int("desc_max_iter", 10, 100)
    poses = met(
        func_n=f,
        start=st,
        strategy=strategy,
        bounds=bnd,
        tol=tol,
        max_iter=desc_max_iter,
    )
    return f(poses[-1])

def make_method_objective(f, sl, met, st=np.array([0, 0]), bnd=None):
    return lambda tr: method_learner(tr, f, sl, met, st, bnd)


# params ===
func = fn.himmelblau
strat_learner = wolfe_strategy_learner
method = ut.gradient_descent
start = np.array([1, 1])
bounds = np.array([[-4, 4], [-4, 4]])
# params ===


objective = make_method_objective(func, strat_learner, method, start, bounds)
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)
print("Best params:", study.best_params)
print("Best value :", study.best_value)
