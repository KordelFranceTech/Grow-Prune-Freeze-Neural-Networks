# Configuration Schema

All experiment configurations live in this directory as YAML files. One file per experiment. Configs are versioned with the code; never edit a config in-place after running an experiment with it — copy and rename.

## File naming

`{component}_{variant}_{date}.yaml`, e.g. `wm_baseline_2025_05.yaml`, `planner_riskaverse_2025_06.yaml`.

The `default.yaml` file is the canonical reference; new configs inherit from it via `base: default.yaml` and override only the fields they change.

## Schema sections

A complete config has five top-level sections, mirroring the project's component structure:

- `simulator` — wind regime, source distribution, sensor model, dataset size
- `tokenization` — bin boundaries, vocabulary version, special token IDs
- `world_model` — architecture, training, inference settings
- `planner` — rollout count, length, scoring weights, replanning frequency
- `validation` — which tests to run, pass/fail thresholds

Not every config needs every section. A "world model only" experiment can omit `planner`. A "planner tuning" experiment that uses a frozen world model checkpoint references that checkpoint by path and overrides only the `planner` section.

## Field types and units

- All distances in meters, all times in seconds, all angles in degrees (not radians) for human readability.
- Concentration bin boundaries are expressed as multiples of the sensor noise floor σ, not in absolute units. This decouples bin definitions from sensor calibration.
- Frequencies in Hz.

## Reproducibility

Every config must include a top-level `seed` field. Simulator data generation, training, and evaluation all derive their RNG state from this seed plus a component-specific salt. A config + the codebase at a given commit reproduces an experiment bit-exactly.

## Required fields for every config

```yaml
name: string                  # human-readable experiment name
seed: int                     # master RNG seed
notes: string                 # one-paragraph description of intent
base: string (optional)       # path to parent config to inherit from
```

## Component sections

See `default.yaml` for the canonical schema with all fields filled in and commented.
