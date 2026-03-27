1. Enter the data
   ---------------------
g - categorical: each value has its own small model
z - continuous: regressor for the given models
y - response variable

3. Pick mixing method
   ---------------------
forecast mixing: optimally weighs forecasts of individual smaller models (best option)
model mixing: mixes models after each observation (could be used with forecast mixing in tandem)
- the optimal mixed models are found either numerically (expensive, high probability of not reaching optima)
- or "analytically": the block structure of optimal V can be analytically found but relies on numerical estimation of nu (reliable, good)

3. RUN LOO (leave-one-out experiment)
   ---------------------
take the optimal structure and run the LOO experiment with it, compare metrics to any benchmark model (mean and OLS benchmarks are included)

4. Find optimal structure
   ---------------------
(based on the log-likelihood of the left out observation -> relies on the hidden observation but helps to eliminate redundant categories)
brute force: searches through all the possible combinations of different g's (very slow: 2^n complexity, finds global optima, feasible for low amount of categories)
greedy search: backward elimination (starts with full g and iteratively removes categories), forward search (starts with empty g and iteratively adds categories)


