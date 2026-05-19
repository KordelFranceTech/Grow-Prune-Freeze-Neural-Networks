# gpf/ — Paper artifacts and reference material

## What belongs here

- `gpf_experiment.tex` — LaTeX drop-in for the Experiments section: plume simulator, sensor model, state representation, GPF agent setup, reward, training protocol, and hyperparameter table
- `gpf_results.tex` — LaTeX drop-in for the Results section: training dynamics, grow/prune event narrative, episode-length ablation finding, and 94/100 generalization evaluation
- `plot_training_curve.py` — standalone script that generates `fig_training_curve.{pdf,png}` from the hardcoded exp11b training data
- `fig_training_curve.pdf` / `fig_training_curve.png` — publication-quality training curve figure (already generated; regenerate with the script if needed)
- `GPFs_reference.pdf` — dissertation reference (Chapter 7, Table 7.1, Eq. 7.4) for GPF parameter notation
- `train_gpt_baseline.py` / `train_gpt_gpf.py` — earlier GPT-based GPF prototype (not the active experiment path)

## Regenerating the figure

```bash
# from project root
python3 gpf/plot_training_curve.py                              # → gpf/fig_training_curve.pdf
python3 gpf/plot_training_curve.py --out gpf/fig_training_curve.png
```

## Including in LaTeX

```latex
\input{gpf/gpf_experiment}   % Experiments section
\input{gpf/gpf_results}      % Results section

% Figure reference used in gpf_results.tex:
\begin{figure}[h]
  \centering
  \includegraphics[width=0.85\textwidth]{gpf/fig_training_curve}
  \caption{GPF Expected SARSA training curve on the plume navigation task.
           Dashed green lines mark grow events; purple annotations show the
           fraction of weights retained by each OBD prune pass.
           The red dot marks the best checkpoint (98\% at episode 3{,}000).
           Layer count $L$ is indicated along the bottom axis.}
  \label{fig:gpf_training_curve}
\end{figure}
```

Required packages: `booktabs` (for the hyperparameter table).

## Key numbers for the paper (exp11b, seed 42)

| Metric | Value |
|--------|-------|
| Best training success rate | 98.0% (ep 3000, 3-layer sparse) |
| Generalization (100 random seeds) | 94 / 100 (94%) |
| Failure mode | All 6 failures were timeouts, not navigational errors |
| Episode budget | 100 s / 1200 steps |
| State space | 392 discrete states (7 × 7 × 8) |
| Input dimension | 22 (one-hot) |
| Best checkpoint architecture | 3 hidden layers, 64-wide, ~48% sparse layer 0 |

## References needed in bibliography

- `farrell2002filament` — Farrell & Murlis (2002), filament plume model
- `lecun1990optimal` — LeCun et al. (1989/1990), Optimal Brain Damage
- `ng1999policy` — Ng et al. (1999), potential-based reward shaping
