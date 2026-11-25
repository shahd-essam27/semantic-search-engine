# Customer-review demonstration corpus

`customer_reviews.csv` contains **126 original synthetic demonstration reviews** across 6 themes: battery, delivery, customer_support, usability, value, and build_quality. These are constructed examples for practice, **not reviews collected from real customers**.

Columns: `id`, `text`, `category` (reference label, not an input to KMeans), `source`.

Six additional paraphrased reviews are included to illustrate near-duplicate detection. A cosine threshold may miss some of them or flag other pairs; review results rather than assuming all intended pairs will be found.

Task 4.1 uses this dataset to demonstrate the pipeline. Replace it with a properly licensed authentic customer-review corpus if your instructor requires genuine reviews.
