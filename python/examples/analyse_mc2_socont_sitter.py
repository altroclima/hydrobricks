import shutil
import matplotlib.pyplot as plt
import spotpy
import hydrobricks as hb
from setups.socont_sitter import parameters, socont, forcing, obs, tmp_dir

# Select the parameters to optimize/analyze
parameters.allow_changing = ['a_snow', 'k_quick', 'A', 'k_slow_1', 'percol', 'k_slow_2',
                             'precip_corr_factor']

# Setup SPOTPY
spot_setup = hb.SpotpySetup(socont, parameters, forcing, obs, warmup=365,
                            obj_func=spotpy.objectivefunctions.nashsutcliffe)

# Select number of runs and run spotpy
nb_runs = 50
sampler = spotpy.algorithms.mc(spot_setup, dbname='socont_sitter_MC', dbformat='csv')
sampler.sample(nb_runs)

# Load the results
results = sampler.getdata()

# Plot parameter interaction
spotpy.analyser.plot_parameterInteraction(results)
plt.tight_layout()
plt.show()

# Plot posterior parameter distribution
posterior = spotpy.analyser.get_posterior(results, percentage=10)
spotpy.analyser.plot_parameterInteraction(posterior)
plt.tight_layout()
plt.show()

# Cleanup
try:
    socont.cleanup()
    shutil.rmtree(tmp_dir)
except Exception:
    print("Failed to clean up.")