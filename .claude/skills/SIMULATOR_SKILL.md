# Simulator Skill

## Purpose

Generate physically realistic plume trajectories for training and evaluation. The simulator is the single source of truth for training data; do not generate training data with an LLM or any other surrogate.

## When to use this skill

Use when:
- Implementing or extending the plume physics simulator
- Generating new training datasets
- Adding new wind conditions, source configurations, or sensor models
- Building evaluation scenarios

Do not use this skill for: tokenization, model architecture, or planner logic.

## Simulator type

Filament-based plume simulator in the spirit of Farrell, Murlis, Long & Cardé (2002). The simulator releases discrete filaments from the source at a configurable rate; each filament advects with the (possibly time-varying, turbulent) wind field and grows by diffusion. The concentration at any point is the sum of contributions from all nearby filaments under a Gaussian kernel proportional to filament age.

This class of simulator is the standard for plume-tracking research because it:
- Reproduces realistic intermittency (whiffs and blanks)
- Captures filament structure that drives bilateral asymmetry
- Runs fast enough to generate large training datasets
- Has well-understood failure modes

Reference: Farrell, J. A., Murlis, J., Long, X., & Cardé, R. T. (2002). "Filament-based atmospheric dispersion model to achieve short time-scale structure of odor plumes."

## Wind model

At minimum, support:
- Constant wind (debugging only)
- Mean wind + Gaussian gusts (mild turbulence)
- Mean wind + Ornstein-Uhlenbeck process for direction and magnitude (realistic)

For training data diversity, sample wind parameters per-trajectory from a distribution covering the deployment range. Document the training wind distribution; the world model will generalize poorly outside it.

## Sensor model

Two antennae at fixed positions relative to the agent body. At each timestep, each antenna samples concentration as the filament-sum at its location plus Gaussian sensor noise. The noise level σ is the same σ used as the unit for concentration bin boundaries — keep them consistent.

Sensor saturation: cap concentration at a configurable maximum (real sensors saturate). The top concentration bin C5 should correspond to "near saturation."

## Training data generation

For each training trajectory:
1. Sample wind parameters from the training distribution
2. Sample source position and strength
3. Sample initial agent position (varied distances and bearings from source)
4. Choose a behavior policy for the agent (see below)
5. Run the simulator forward, logging at 10 Hz: agent position, agent heading, left antenna concentration, right antenna concentration, wind direction, action taken
6. Tokenize and save

**Behavior policy diversity matters.** If all training trajectories are generated with one policy, the world model only learns plume dynamics under that policy's state distribution. Use a mix:
- Random walk (broad coverage)
- Reactive (turn upwind on whiff, cast on blank)
- Infotaxis or surge-and-cast (closer to optimal)
- The planner itself, once it works (DAgger-style iterative data collection)

Aim for at least 10⁶ timesteps total, balanced across policies.

## Evaluation scenarios

Separate from training data, define a fixed evaluation suite of scenarios:
- Easy: steady wind, agent starts inside the plume cone, source 5m away
- Medium: moderate turbulence, agent starts at edge of plume, source 10m away
- Hard: high turbulence, agent starts outside plume, must search
- Out-of-distribution: wind regimes not seen in training

Each scenario is a fixed seed + config. Evaluation reports success rate (source found within time budget) and time-to-source per scenario.

## Common pitfalls

- **Constant wind in training.** Models trained on constant wind fail catastrophically in turbulence. Always include turbulent conditions in training.
- **Single agent policy in training.** See above — diversify.
- **Ignoring sensor noise.** Real sensors are noisy; training without noise produces a model that's overconfident in low-concentration regions.
- **Tokenizing on the fly during training.** Tokenize once at dataset generation time and store tokenized trajectories. Avoids re-tokenization bugs and speeds up training.

## Files in this directory

- `filament_sim.py` — core simulator
- `wind.py` — wind models
- `sensors.py` — antenna sensor model
- `policies.py` — behavior policies for training data generation
- `generate_dataset.py` — dataset generation script
- `scenarios/` — evaluation scenario configs
