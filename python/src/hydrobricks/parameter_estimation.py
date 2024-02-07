import importlib
import os
from datetime import datetime

import numpy as np

import hydrobricks as hb


class SpotpySetup:

    def __init__(self, model, params, forcing, obs, warmup=365, obj_func=None,
                 invert_obj_func=False, dump_outputs=False, dump_forcing=False,
                 dump_dir='', combine_multicalib='mean'):
        
        if type(model) is list:
            if len(model) != len(forcing) or len(model) != len(obs):
                raise RuntimeError('The model, forcing and observation lists have different lengths.')
            self.model = model
            self.forcing = forcing
            self.obs = [o.data[0] for o in obs]
        else:
            self.model = [model]
            self.forcing = [forcing]
            self.obs = [obs.data[0]]
        self.params = params
        self.params_spotpy = params.get_for_spotpy()
        self.random_forcing = params.needs_random_forcing()
        self.forcing[:].apply_operations(params)
        self.warmup = warmup
        self.obj_func = obj_func
        self.invert_obj_func = invert_obj_func
        self.dump_outputs = dump_outputs
        self.dump_forcing = dump_forcing
        self.dump_dir = dump_dir
        self.combined_multicalib = combined_multicalib
        if not self.random_forcing:
            for m, f in zip(self.model, self.forcing):
                m.set_forcing(forcing=f)
        if not obj_func:
            print("Objective function: Non parametric Kling-Gupta Efficiency.")

    def parameters(self):
        x = None
        for i in range(1000):
            x = hb.spotpy.parameter.generate(self.params_spotpy)
            names = [row[1] for row in x]
            values = [row[0] for row in x]
            params = self.params
            param_values = dict(zip(names, values))
            params.set_values(param_values, check_range=False)

            if params.constraints_satisfied() and params.range_satisfied():
                break

            if i >= 1000:
                raise RuntimeError('The parameter constraints could not be satisfied.')

        return x
    
    def individual_simulation(self, model, forcing, forcing_filename, params):
        if self.random_forcing:
            forcing.apply_operations(params, apply_to_all=False)
            model.run(parameters=params, forcing=forcing)
        else:
            model.run(parameters=params)
        sim = model.get_outlet_discharge()

        if self.dump_outputs or self.dump_forcing:
            now = datetime.now()
            date_time = now.strftime("%Y-%m-%d_%H%M%S")
            path = os.path.join(self.dump_dir, date_time)
            os.makedirs(path, exist_ok=True)
            if self.dump_outputs:
                model.dump_outputs(path)
            if self.dump_forcing:
                forcing.save_as(os.path.join(path, forcing_filename))
        return sim[self.warmup:]

    def simulation(self, x):
        params = self.params
        param_values = dict(zip(x.name, x.random))
        params.set_values(param_values)

        if not params.constraints_satisfied() or not params.range_satisfied():
            return np.random.rand(len(self.obs[0][self.warmup:]))

        sims = []
        for i, (model, forcing) in enumerate(zip(self.model, self.forcing)):
            forcing_filename = f'forcing_{i}.nc'
            if len(self.model) == 0:
                forcing_filename = f'forcing.nc'
            sim = individual_simulation(model, forcing, forcing_filename, params)
            sims.append(sim)

        return sims

    def evaluation(self):
        obs = [o[self.warmup:] for o in self.obs]
        return obs

    def objectivefunction(self, simulation, evaluation, params=None):
        likes = []
        for simu, eval in zip(simulation, evaluation):
            if not self.obj_func:
                like = hb.spotpy.objectivefunctions.kge_non_parametric(eval,
                                                                       simu)
            elif isinstance(self.obj_func, str):
                like = hb.evaluate(simu, eval, self.obj_func)
            else:
                like = self.obj_func(eval, simu)
    
            if self.invert_obj_func:
                like = -like
            likes.append(like)

        if len(self.model) == 0:
            likes = likes[0]
        elif len(self.model) > 1:
            if self.combine_multicalib == 'mean':
                likes = sum(likes) / len(likes) 

        return likes


def evaluate(simulation, observation, metric):
    """
    Evaluate the simulation using the provided metric (goodness of fit).

    Parameters
    ----------
    simulation : np.array
        The predicted time series.
    observation : np.array
        The time series of the observations with dates matching the simulated
        series.
    metric : str
        The abbreviation of the function as defined in HydroErr
        (https://hydroerr.readthedocs.io/en/stable/list_of_metrics.html)
        Examples: nse, kge_2012, ...

    Returns
    -------
    The value of the selected metric.
    """
    eval_fct = getattr(importlib.import_module('HydroErr'), metric)

    return eval_fct(simulation, observation)
