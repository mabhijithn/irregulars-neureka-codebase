# Wiener pre-processing
Wiener pre-processing builds a filter bank of spatio-temporal filters based on artifacts identified in the training set.

1. A set of high power non-seizure epochs are identified
2. PCA compressed spatio-temporal covariance matrices are used to represent the artifacts
3. K-means clustering is used to group the artifacts
4. The average representation of the groups with the most artifacts is used to pre-compute a spatio-temporal wiener filter
