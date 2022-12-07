import shutil
import matplotlib.pyplot as plt
from setups.socont_sitter import parameters, socont, forcing, obs, tmp_dir

# Select the parameters to optimize/analyze
parameters.allow_changing = ['a_snow', 'k_quick', 'A', 'k_slow_1', 'percol', 'k_slow_2',
                             'precip_corr_factor']

# Proceed to the Monte Carlo analysis
mc_analysis = socont.analyze(
    method='monte_carlo', metrics=['kge_2012', 'nse', 'me'], forcing=forcing,
    parameters=parameters, nb_runs=1000, observations=obs.data_raw[0])

# Plot
for param in parameters.allow_changing:
    mc_analysis.plot.scatter(x=param, y='nse')
    plt.ylim(0, 1)
    plt.title(param)
    plt.tight_layout()
    plt.show()

# Cleanup
try:
    socont.cleanup()
    shutil.rmtree(tmp_dir)
except Exception:
    print("Failed to clean up.")