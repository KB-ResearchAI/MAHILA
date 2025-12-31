\# Differences from Image-Only Chronological Models



This document enumerates the \*\*only changes\*\* relative to the image-only

multi-image models.





\## Unchanged

\- Dataset

\- Chronological split

\- Model architectures

\- Backbone networks

\- Hyperparameters

\- Training procedure

\- Evaluation metrics

\- Cross-validation protocol





\## Changed

\- Addition of maternal metadata:

&nbsp; - Age at registration

&nbsp; - Days since LMP

&nbsp; - Lighting indicators

\- Metadata concatenated at feature fusion layer





\## Not Changed

\- No architectural modifications





\## Rationale

The minimal-change design ensures that any performance difference is

attributable solely to metadata inclusion.



Observed results indicate negligible incremental benefit from metadata.



