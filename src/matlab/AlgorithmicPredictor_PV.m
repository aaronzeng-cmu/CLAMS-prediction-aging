function [Pred, state] = AlgorithmicPredictor_PV(chunk, state, shift_ms, fs)
% AlgorithmicPredictor_PV  Phase extrapolation via dominant SO-band FFT peak.
%
% Identifies the dominant frequency in the SO band (0.4–1.2 Hz) within a
% Hann-windowed FFT buffer, then extrapolates the phase by
%   phi_pred = angle(FFT[k_dom]) + 2*pi*f_dom*(shift_ms/1000)
%
% Inputs:
%   chunk     [N x 1] — new raw EEG samples
%   state     struct with fields:
%               .buffer  [N_fft x 1] — FFT buffer (default: 512 samples)
%               .bp_lo   scalar — SO band lower edge (Hz)
%               .bp_hi   scalar — SO band upper edge (Hz)
%               .N_fft   scalar — FFT buffer length
%   shift_ms  scalar — prediction horizon (ms)
%   fs        scalar — sample rate (Hz)
%
% Outputs:
%   Pred   {1}[2 x N] — [cos(phi_pred); sin(phi_pred)] repeated across chunk
%   state  updated state struct

N = length(chunk);

% --- Initialize state ---
if ~isfield(state, 'buffer') || isempty(state.buffer)
    state.N_fft  = 512;
    state.bp_lo  = 0.4;
    state.bp_hi  = 1.2;
    state.buffer = zeros(state.N_fft, 1);
end

% --- Update rolling buffer ---
if N >= state.N_fft
    state.buffer = chunk(end - state.N_fft + 1:end);
else
    state.buffer = [state.buffer(N+1:end); chunk(:)];
end

buf    = state.buffer;
N_fft  = state.N_fft;

% --- Hann-windowed FFT ---
win    = hann(N_fft);
X_fft  = fft(buf .* win, N_fft);

% Frequency axis
freqs  = (0:N_fft-1) * (fs / N_fft);

% --- Find dominant bin in SO band ---
so_mask = (freqs >= state.bp_lo) & (freqs <= state.bp_hi);
so_mask(1) = false;    % exclude DC

if ~any(so_mask)
    % Band too narrow for current fs/N_fft — fallback to 0.75 Hz estimate
    f_dom   = 0.75;
    phi_est = 0;
else
    power   = abs(X_fft) .^ 2;
    power_so = power;
    power_so(~so_mask) = 0;
    [~, k_dom] = max(power_so);

    f_dom   = freqs(k_dom);
    phi_est = angle(X_fft(k_dom));
end

% --- Phase extrapolation ---
phi_pred = phi_est + 2 * pi * f_dom * (shift_ms / 1000);

% Wrap to [-pi, pi]
phi_pred = atan2(sin(phi_pred), cos(phi_pred));

% --- Return [cos; sin] replicated across chunk ---
cs   = [cos(phi_pred); sin(phi_pred)];
Pred = {repmat(cs, 1, N)};

end
