# E1 reuse compatibility check (2026-11-10)

FACT: On local main at 1b5030b, the diff from E1 execution commit ddfb650 in em_runner.py, model_dual_expfam_consistent.py, and laplace_k_criterion.py contains only E2a's optional init_seed/mc_seed split in em_runner.py. The default for both is None; the code then uses the original seed for both initialization and MC draws. MX2 does not pass either optional argument. The model-consistent and joint-mode files have no diff in this comparison.

FACT: E1's full/reproduction.json records exact reproduction of 480 M1 scalar results, with maximum absolute difference 0.0. MX2's protocol.json records SHA-256 hashes for the two E1 source CSVs, and mask_provenance.csv verifies dataset and M1 dense/sparse mask hashes for every replicate. Reused rows retain source=E1/full and are never passed to MCEM again in MX2.

LIMIT: Shared seed and default-path source equivalence support the reuse, but the new Gaussian arms have distinct family-dependent computations and are not assumed to have identical internal random streams. The E1 source CSV does not include held-out AP; this cell is left empty.
