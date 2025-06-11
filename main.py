from sdt_ddm import read_data, apply_hierarchical_sdt_model, draw_delta_plots
import pymc as pm
import arviz as az

# Load SDT-formatted data
sdt_data = read_data("data.csv", prepare_for="sdt", display=False)

# Build the hierarchical SDT model
sdt_model = apply_hierarchical_sdt_model(sdt_data)
print("\nModel built successfully.")

# Sample from the model
with sdt_model:
    trace = pm.sample(draws=1000, tune=1000, target_accept=0.9, return_inferencedata=True)

# Save and summarize posterior samples
az.to_netcdf(trace, "sdt_trace.nc")
print("\nSampling complete. Posterior samples saved to 'sdt_trace.nc'.")

trace = az.from_netcdf("sdt_trace.nc")
summary = az.summary(trace, var_names=["mean_d_prime", "mean_criterion"], round_to=2)
print("\nPosterior Summary (mean_d_prime & mean_criterion):")
print(summary)

# Generate delta plots
delta_data = read_data("data.csv", prepare_for="delta plots", display=False)
draw_delta_plots(delta_data, pnum=1)



