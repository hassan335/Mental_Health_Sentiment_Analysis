# Data Preparation Summary

Task:                   Classification
Data original shape:    26264 samples × 770 features
Data final shape:       26264 samples × 768 features
Target feature:         label

Samples dropped due to NaN target: 0
Indicator variables added for continuous NaNs: 0

# Processing Times

| computation          |   runtime (ms) |
|:---------------------|---------------:|
| unify_nans           |          28172 |
| convert_categoricals |            134 |
| inspect_target       |             89 |
| drop_target_nans     |            142 |
| encode_target        |            187 |
| drop_unusable        |          38242 |
| deflate_categoricals |             21 |
| encode_categoricals  |         141119 |