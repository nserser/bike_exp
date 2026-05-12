# GAP Items

- GAP-01: Production-parameter certification is not implemented. The current
  artifact targets reduced domains and sampled-key diagnostics only.
- GAP-02: Analytic screened-binomial dirty-tail certificates are not the primary
  v01 path. Enumeration-first transition envelopes remain the planned route.
- GAP-03: Full near-codeword family enumeration is not implemented at larger
  parameters. v01 is expected to use canonical rotations and optional bounded
  search.
- GAP-04: State compression may omit dependencies that matter at later
  iterations. Looseness and ablation reporting will be used to diagnose this.
- GAP-05: Sampled-key frequency envelopes with rigorous confidence bounds are
  optional in v01. Larger sampled runs must be labeled accordingly unless added.
- GAP-06: DOCX/shared-object storage-size optimization is irrelevant to this
  artifact, but the project-level open item remains acknowledged.
- GAP-07: Medium-size scripts now exist for r=61 and r=89, but their shipped
  main-run threshold schedules are defaults. Each main run should be preceded by
  its sweep and updated if the sweep selects a different schedule.
- GAP-08: The r=127,d=11 diagnostic is profile-only. It should not be reported
  as a DFR estimate or certification result.

## Step missing-size wrapper update

The missing-size wrapper previously defaulted to `sweeps`, so a bare invocation
produced calibration artifacts but did not run the full same-key model
comparisons. The wrapper now defaults to `main`, with `sweeps`, `profiles`,
`selected`, and `full/all` modes documented. Main configs were updated to the
sweep-selected schedules observed in the missing-size calibration results:
`r61,d7,t2 -> 5;4;4;3`, `r89,d9,t2 -> 5;5;4;3`, and
`r89,d9,t3 -> 6;5;5;4;3`.
