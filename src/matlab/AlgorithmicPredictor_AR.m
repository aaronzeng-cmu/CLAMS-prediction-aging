function [Pred, state] = AlgorithmicPredictor_AR(chunk, state, shift_ms, fs)
% AlgorithmicPredictor_AR  Autoregressive (Yule-Walker) SO phase predictor.
%
% Maintains a rolling buffer of past EEG samples, fits an AR(p) model,
% extrapolates forward by shift_samples, then bandpass-filters and applies
% the Hilbert transform to recover instantaneous phase.
%
% Inputs:
%   chunk     [N x 1] — new raw EEG samples (single chunk)
%   state     struct with fields (initialized on first call):
%               .buffer  [buf_len x 1] — circular sample buffer
%               .bp_lo   scalar — bandpass lower edge (Hz), default 0.4
%               .bp_hi   scalar — bandpass upper edge (Hz), default 1.2
%               .ar_p    scalar — AR order, default 200
%               .buf_len scalar — buffer length in samples, default 5*fs
%   shift_ms  scalar — prediction horizon (ms)
%   fs        scalar — sample rate (Hz)
%
% Outputs:
%   Pred   {1}[2 x N] — [cos(phi); sin(phi)] repeated across chunk (same phase)
%   state  updated state struct

N             = length(chunk);
shift_samples = round(shift_ms * fs / 1000);

% --- Initialize state on first call ---
if ~isfield(state, 'buffer') || isempty(state.buffer)
    state.bp_lo  = 0.4;
    state.bp_hi  = 1.2;
    state.ar_p   = min(200, round(fs / 0.75));   % ≈ 1 cycle of SO at fs
    state.buf_len = 5 * fs;
    state.buffer  = zeros(state.buf_len, 1);
end

% --- Update circular buffer ---
state.buffer = [state.buffer(N+1:end); chunk(:)];

buf = state.buffer;

% --- Fit AR model ---
% aryule requires Signal Processing Toolbox
try
    [ar_coeffs, ~] = aryule(buf, state.ar_p);
catch
    % Fallback: return zero-phase prediction if toolbox unavailable
    phi_pred = 0;
    cs = [cos(phi_pred); sin(phi_pred)];
    Pred = {repmat(cs, 1, N)};
    return
end

% --- Forward predict shift_samples into the future ---
predicted = double(buf);
for s = 1:shift_samples
    % AR prediction: next = -sum(ar(2:end) .* predicted(end-p+1:end))
    x_win = predicted(end - state.ar_p + 1:end);
    next_val = -ar_coeffs(2:end) * flipud(x_win);
    predicted = [predicted; next_val];
end

% --- Bandpass + Hilbert on extended signal ---
[b_bp, a_bp] = butter(4, [state.bp_lo, state.bp_hi] / (fs/2), 'bandpass');
bp_signal = filtfilt(b_bp, a_bp, predicted);
analytic  = hilbert(bp_signal);

% Phase at the predicted future point (end of extended signal)
phi_pred = angle(analytic(end));

% --- Return [cos; sin] replicated across chunk timesteps ---
cs   = [cos(phi_pred); sin(phi_pred)];
Pred = {repmat(cs, 1, N)};

end
