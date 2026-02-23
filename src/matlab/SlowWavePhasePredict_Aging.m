function [vars, Graph, EEG] = SlowWavePhasePredict_Aging(EEG, vars, Graph)
% SlowWavePhasePredict_Aging  Drop-in replacement for SlowWavePhasePredict.m
%
% Same three-argument signature (EEG, vars, Graph) for compatibility with the
% EEG-LLAMAS pipeline. Supports neural (LSTM, TCN) and algorithmic (AR, PV,
% SSPE) phase predictors through a unified routing layer.
%
% Required vars fields (set in the CLAMS initialization script):
%   vars.ModelType     — 'lstm' | 'tcn' | 'ar' | 'pv' | 'sspe'
%   vars.ModelPath     — path to .mat (lstm) or .onnx (tcn) file
%   vars.FeatureConfig — struct:
%                          .fs       sample rate (Hz), default 200
%                          .bp_lo    bandpass lo (Hz), default 0.4
%                          .bp_hi    bandpass hi (Hz), default 1.2
%                          .bp_order filter order, default 4
%                          .sspe_f0  [optional] SSPE center freq (Hz)
%   vars.ShiftMs       — prediction horizon (ms), e.g. 100
%   vars.DeltaThresh   — delta power threshold (µV²), same as reference
%   vars.MagThresh     — minimum predictor magnitude (e.g. 0.5)
%   vars.PhaseWindow   — [lo_deg, hi_deg] trigger phase window, default [-60, -10]
%   vars.RefractoryMs  — minimum inter-stimulus interval (ms), default 2000
%
% Persistent state fields written by this function:
%   vars.PhasePredictor — model state (dlnetwork or algorithmic struct)
%   vars.z              — IIR feature filter state
%   vars.b, vars.a      — IIR filter coefficients
%   vars.prev_sample    — last sample from previous chunk
%   vars.refractory_ctr — refractory counter (samples)
%   vars.algo_state     — algorithmic predictor state (AR/PV/SSPE)
%   vars._initialized   — logical, true after first-call setup

% =========================================================================
% 0. Lazy initialization on first call
% =========================================================================
if ~isfield(vars, '_initialized') || ~vars.('_initialized')
    vars = init_predictor(vars);
end

% =========================================================================
% 1. Extract EEG chunk
% =========================================================================
[chunk, EEG] = get_eeg_chunk(EEG, vars);
N = length(chunk);

if N == 0
    return
end

% =========================================================================
% 2. Build features (needed for neural models)
% =========================================================================
sample_with_prev = [vars.prev_sample; chunk(:)];
vars.prev_sample = chunk(end);

% =========================================================================
% 3. Route by model type
% =========================================================================
model_type = lower(vars.ModelType);

switch model_type

    % --- LSTM ------------------------------------------------------------
    case 'lstm'
        [X, vars] = ExtractFeatures_Aging(sample_with_prev, vars);
        [vars.PhasePredictor, Pred] = predictAndUpdateState(vars.PhasePredictor, {X});

    % --- TCN (circular buffer) -------------------------------------------
    case 'tcn'
        [X_new, vars] = ExtractFeatures_Aging(sample_with_prev, vars);
        % X_new: [C x N x 1 x 1] -> [C x N]
        new_features = squeeze(X_new);                 % [C x N]
        if isvector(new_features) && size(X_new,2) == 1
            new_features = reshape(new_features, size(X_new,1), 1);
        end
        rf = vars.PhasePredictor.rf;
        buf = vars.PhasePredictor.buffer;               % [C x rf]
        buf = [buf(:, N+1:end), single(new_features)]; % shift left, append right
        vars.PhasePredictor.buffer = buf;

        dlBuf = dlarray(buf, 'CT');
        Pred_raw = predict(vars.PhasePredictor.net, dlBuf);  % [2 x rf]
        Pred_raw_d = double(extractdata(Pred_raw));
        % Use last column; replicate across chunk timesteps
        last_col = Pred_raw_d(:, end);
        Pred = {repmat(last_col, 1, N)};                % {[2 x N]}

    % --- AR (Yule-Walker) ------------------------------------------------
    case 'ar'
        [Pred, vars.algo_state] = AlgorithmicPredictor_AR( ...
            chunk, vars.algo_state, vars.ShiftMs, vars.FeatureConfig.fs);

    % --- PV (Peak FFT Velocity) ------------------------------------------
    case 'pv'
        [Pred, vars.algo_state] = AlgorithmicPredictor_PV( ...
            chunk, vars.algo_state, vars.ShiftMs, vars.FeatureConfig.fs);

    % --- SSPE (State-Space Phase Estimation) -----------------------------
    case 'sspe'
        if isfield(vars.FeatureConfig, 'sspe_f0') && ~isempty(vars.FeatureConfig.sspe_f0)
            vars.algo_state.sspe_f0 = vars.FeatureConfig.sspe_f0;
        end
        [Pred, vars.algo_state] = AlgorithmicPredictor_SSPE( ...
            chunk, vars.algo_state, vars.ShiftMs, vars.FeatureConfig.fs);

    otherwise
        error('SlowWavePhasePredict_Aging:unknownModelType', ...
            'Unknown ModelType ''%s''. Use lstm|tcn|ar|pv|sspe.', vars.ModelType);
end

% ALL paths now have Pred = {[2 x N]}

% =========================================================================
% 4. Decode phase from last timestep output
% =========================================================================
PredAngle = angle(Pred{1}(1,end) + sqrt(-1) * Pred{1}(2,end));
Mag       = norm(Pred{1}(:,end));

% =========================================================================
% 5. Apply gating (shared across all model types)
% =========================================================================
[Graph, vars] = apply_gating(PredAngle, Mag, EEG, vars, Graph);

end  % main function


% =========================================================================
% LOCAL: init_predictor
% =========================================================================
function vars = init_predictor(vars)
% Lazy initialization: called once on first chunk.

fs = vars.FeatureConfig.fs;

% --- Design IIR bandpass filter (BA form for MATLAB filter()) ---
[b, a] = butter(vars.FeatureConfig.bp_order, ...
    [vars.FeatureConfig.bp_lo, vars.FeatureConfig.bp_hi] / (fs/2), ...
    'bandpass');
vars.b = b;
vars.a = a;
vars.z = zeros(length(a)-1, 1);   % zero-init filter state

vars.prev_sample = 0;

% --- Load or init predictor ---
model_type = lower(vars.ModelType);
switch model_type
    case {'lstm', 'tcn'}
        vars.PhasePredictor = LoadAgingPredictor( ...
            vars.ModelPath, vars.ModelType, 'shift_ms', vars.ShiftMs);
    case {'ar', 'pv', 'sspe'}
        vars.algo_state = struct();
    otherwise
        error('init_predictor:unknownType', 'Unknown ModelType: %s', vars.ModelType);
end

% --- Default gating parameters if not set ---
if ~isfield(vars, 'PhaseWindow'),   vars.PhaseWindow   = [-60, -10]; end
if ~isfield(vars, 'MagThresh'),     vars.MagThresh     = 0.5;        end
if ~isfield(vars, 'RefractoryMs'),  vars.RefractoryMs  = 2000;       end
if ~isfield(vars, 'DeltaThresh'),   vars.DeltaThresh   = 0;          end

vars.refractory_ctr = 0;
vars.('_initialized') = true;

end  % init_predictor


% =========================================================================
% LOCAL: get_eeg_chunk
% =========================================================================
function [chunk, EEG] = get_eeg_chunk(EEG, vars)
% Extract the current EEG chunk. Mirrors reference SlowWavePhasePredict.m
% boundary handling.

if isfield(EEG, 'data') && ~isempty(EEG.data)
    % EEGLAB-style: EEG.data is [channels x samples]
    ch = 1;  % primary channel
    chunk = double(EEG.data(ch, :))';   % [N x 1]
else
    chunk = [];
end

end  % get_eeg_chunk


% =========================================================================
% LOCAL: apply_gating
% =========================================================================
function [Graph, vars] = apply_gating(PredAngle, Mag, EEG, vars, Graph)
% Shared gating logic for all model types.
%
% Triggers a stimulation marker if ALL conditions are met:
%   1. Predicted phase in PhaseWindow (degrees)
%   2. Predictor magnitude >= MagThresh
%   3. Not in refractory period
%
% Phase window is specified in degrees (matching reference CLAMS convention)
% and converted to radians internally.

fs = vars.FeatureConfig.fs;

% Decrement refractory counter
if vars.refractory_ctr > 0
    vars.refractory_ctr = vars.refractory_ctr - 1;
end

% Convert phase window from degrees to radians
pw_lo = vars.PhaseWindow(1) * pi / 180;
pw_hi = vars.PhaseWindow(2) * pi / 180;

in_phase_window   = (PredAngle >= pw_lo) && (PredAngle <= pw_hi);
mag_ok            = (Mag >= vars.MagThresh);
not_refractory    = (vars.refractory_ctr == 0);

% Combine gating conditions
trigger = in_phase_window && mag_ok && not_refractory;

if trigger
    % Set refractory counter
    vars.refractory_ctr = round(vars.RefractoryMs * fs / 1000);

    % Write trigger marker to Graph (same convention as reference)
    if isfield(Graph, 'trigger')
        Graph.trigger = true;
    end
    if isfield(Graph, 'PredAngle')
        Graph.PredAngle = PredAngle;
    end
end

end  % apply_gating
