# CLAMS Prediction — Aging

**Closed-Loop Auditory Memory Stimulation (CLAMS) — Slow Wave Phase Predictor for Aging Subjects**

## Background

Slow oscillations (SO, ~0.5–1 Hz) during NREM sleep are critical for memory consolidation. Closed-loop auditory stimulation (CLAS) delivers sounds phase-locked to the SO peak (up-state) to enhance slow wave activity (SWA). The EEG-LLAMAS system implements this in real time using an LSTM-based phase predictor trained on young-adult EEG.

Aging significantly alters SO morphology: amplitude decreases, frequency slows, and spatial distribution shifts. A predictor trained on young subjects generalizes poorly to older adults, degrading stimulation precision and potentially limiting the therapeutic benefit of CLAS in this population.

## Project Goal

Train and validate a new SO phase prediction model **specifically for aging subjects**, suitable for drop-in replacement of the existing LSTM in the EEG-LLAMAS closed-loop pipeline.

### Hard Constraints

| Constraint | Rationale |
|---|---|
| MATLAB-deployable | Must run inside EEG-LLAMAS (`predictAndUpdateState` API or equivalent) |
| Causal / online inference | Chunk-by-chunk processing; no future data |
| Compute budget ≈ existing LSTM | Real-time at 200 Hz; latency budget ~5–10 ms per chunk |
| Output: SO phase (radians) | Or equivalent 2D complex representation (cos θ, sin θ) |
| Advance prediction window: 0–110 ms | To compensate hardware + software delays at stimulation delivery |

### Scientific Goal

Predict the instantaneous phase of the bandpass-filtered (0.4–1.2 Hz) slow oscillation from incoming EEG so that auditory stimuli can be triggered at the SO peak (phase ≈ 0 rad), maximizing up-state entrainment in older adults.

## Approach

### Reference Model (EEG-LLAMAS)

The existing model is a **sequence-to-sequence LSTM** (`results.net`) that:

- Runs at **200 Hz** sampling rate
- Accepts 3 input features per sample:
  1. Raw EEG (primary channel)
  2. First difference of raw EEG
  3. Bandpass-filtered EEG (0.4–1.2 Hz Butterworth IIR)
- Outputs a 2D vector `[cos θ, sin θ]` from which phase is recovered: `angle(out(1) + i·out(2))`
- Uses `predictAndUpdateState` for stateful, causal online prediction
- Was trained on **young-adult** EEG (N=2–3 subjects)
- Separate models stored per advance shift (0, 5, 10, ..., 105 ms)

### New Model Development Plan

1. **Data preparation** — preprocess aging NREM EEG; extract slow-wave epochs and phase labels via Hilbert transform on offline-filtered signal
2. **Feature engineering** — replicate existing 3-feature scheme; explore additional features (e.g., spindle-band envelope, multi-channel spatial filtering)
3. **Model training (Python/PyTorch or MATLAB)** — cross-validated training with leave-one-subject-out or k-fold; optimize for circular phase error
4. **MATLAB export** — export trained network to MATLAB (`exportNetworkToONNX` or native MATLAB Deep Learning Toolbox format)
5. **Integration & validation** — offline replay in EEG-LLAMAS pipeline; compare phase-locking value (PLV) and mean absolute phase error (MAPE) vs. reference model

### Candidate Model Architectures

| Architecture | Rationale | MATLAB Support |
|---|---|---|
| LSTM (same as reference) | Proven; direct replacement | Native |
| Temporal Convolutional Net (TCN) | Faster inference; fixed receptive field | Via ONNX |
| Kalman filter + state-space | Interpretable; very low compute | Native |
| WaveNet-lite | Strong temporal modeling | Via ONNX |

## Repository Structure

```
CLAMS_prediction_aging/
├── README.md
├── .gitignore
├── data/                   # Raw & processed EEG (not tracked)
│   ├── raw/
│   └── processed/
├── src/
│   ├── training/           # Model training scripts (Python)
│   ├── evaluation/         # Offline evaluation / phase accuracy metrics
│   └── matlab/             # MATLAB integration: predictor wrapper, postprocessing function
└── models/                 # Exported model files (.mat, .onnx)
```

## Evaluation Metrics

- **Mean Absolute Phase Error (MAPE)** — primary metric; circular distance between predicted and true SO phase at stimulation time
- **Phase-Locking Value (PLV)** — consistency of stimulation relative to true SO phase across trials
- **Computational latency** — wall-clock time per chunk at 200 Hz on target hardware

## References

- Levitt et al. (2023). EEG-LLAMAS. [bioRxiv](https://www.biorxiv.org/content/10.1101/2022.11.21.515651v2)
- Helfrich et al. (2018). Bidirectional prefrontal-hippocampal dynamics organize information transfer during sleep for memory consolidation. *Nature Communications*.
- Mander et al. (2017). Sleep and human aging. *Neuron*.
