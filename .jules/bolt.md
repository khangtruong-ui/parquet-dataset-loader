## 2024-10-14 - PyArrow Contiguous Slicing
**Learning:** For columnar data stores like Parquet accessed via PyArrow, O(N) single-row extraction (using `to_pylist()[0]`) is a massive bottleneck.
**Action:** When a continuous unshuffled slice is requested, always map it to global byte offsets and slice the entire PyArrow Table in bulk (`table.slice()`). This drops slice extraction time for 10k rows from ~0.28s to ~0.002s.
