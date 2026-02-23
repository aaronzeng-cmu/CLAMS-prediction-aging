function [X, vars] = ExtractFeatures_Aging(sample_with_prev, vars)
% ExtractFeatures_Aging  Causal feature extraction for aging SO phase predictor.
%
% Mirrors the Python extract_features() / CausalBandpassFilter.apply() pipeline.
% The caller must prepend the last sample of the previous chunk to ensure
% a continuous first-difference across chunk boundaries.
%
% Inputs:
%   sample_with_prev  [N+1 x 1] — previous sample + N new samples
%   vars              struct with fields:
%                       .b  [1 x (order+1)] — bandpass filter numerator (BA form)
%                       .a  [1 x (order+1)] — bandpass filter denominator (BA form)
%                       .z  [(order) x 1]   — causal filter state (zi from filter())
%
% Outputs:
%   X     [C x N x 1 x 1] — feature array for dlnetwork / predictAndUpdateState
%                           C=3: [raw; diff; bandpass]
%   vars  updated struct with .z carrying the new filter state
%
% Notes:
%   - .z size = length(vars.a) - 1 = filter order (initial value: zeros)
%   - Zero-initialize .z at session start (not steady-state) — EEG has arbitrary DC
%   - For 4th-order Butterworth bandpass: .z is a 4x1 zero vector initially

N   = length(sample_with_prev) - 1;
raw = sample_with_prev(2:end);      % [N x 1]

% Feature 1: raw EEG
feat_raw = raw;

% Feature 2: first difference (requires previous sample)
feat_diff = diff(double(sample_with_prev));  % [N x 1]

% Feature 3: causal bandpass filter (state carried across chunks)
[feat_bp, vars.z] = filter(vars.b, vars.a, double(raw), vars.z);  % [N x 1]

% Stack into [C x N x 1 x 1]
X = zeros(3, N, 1, 1, 'single');
X(1, :, 1, 1) = single(feat_raw);
X(2, :, 1, 1) = single(feat_diff);
X(3, :, 1, 1) = single(feat_bp);

end
