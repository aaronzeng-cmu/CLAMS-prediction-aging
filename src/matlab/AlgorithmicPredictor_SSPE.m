function [Pred, state] = AlgorithmicPredictor_SSPE(chunk, state, shift_ms, fs)
% AlgorithmicPredictor_SSPE  State-Space Phase Estimation (Agarwal et al. 2021).
%
% Implements a Matsuda-Komaki damped oscillator Kalman filter for SO phase
% tracking. The state vector encodes a 2-D oscillator; phase is atan2(x2, x1).
%
% Default aging parameter: f0 = 0.75 Hz (vs ~1.0 Hz for young adults),
% based on El Kanbi 2024 (-32% SO frequency in aged subjects).
%
% State-space model:
%   x(t) = A * x(t-1) + w(t),    w ~ N(0, Q)
%   y(t) = C * x(t) + v(t),      v ~ N(0, R)
%
%   A = rho * [cos(2*pi*f0/fs), -sin(2*pi*f0/fs);
%              sin(2*pi*f0/fs),  cos(2*pi*f0/fs)]
%   C = [1, 0]
%
% Inputs:
%   chunk     [N x 1] — new raw EEG samples (bandpass-filtered externally if needed)
%   state     struct with fields:
%               .x_hat  [2 x 1] — posterior state estimate
%               .P      [2 x 2] — posterior covariance
%               .f0     scalar  — SO center frequency (Hz), default 0.75
%               .rho    scalar  — damping factor in (0,1), default 0.99
%               .sigma_w scalar — process noise std, default 1e-3
%               .sigma_v scalar — observation noise std, default 1.0
%               .n_obs  scalar  — observation count (for init check)
%               .initialized logical
%   shift_ms  scalar — prediction horizon (ms)
%   fs        scalar — sample rate (Hz)
%
% Outputs:
%   Pred   {1}[2 x N] — [cos(phi); sin(phi)] at each sample in chunk
%   state  updated state struct (posterior after processing all samples)
%
% Reference:
%   Agarwal S. et al. (2021). "State space models for real-time phase
%   estimation from EEG." J Neural Eng.

N             = length(chunk);
shift_samples = round(shift_ms * fs / 1000);

% --- Initialize state on first call ---
if ~isfield(state, 'initialized') || ~state.initialized
    state.f0          = 0.75;    % aging default (El Kanbi 2024)
    state.rho         = 0.99;
    state.sigma_w     = 1e-3;
    state.sigma_v     = 1.0;
    state.x_hat       = zeros(2, 1);
    state.P           = eye(2);
    state.n_obs       = 0;
    state.initialized = false;
end

% Optionally expose f0 via vars.FeatureConfig.sspe_f0
if isfield(state, 'sspe_f0') && ~isempty(state.sspe_f0)
    state.f0 = state.sspe_f0;
end

% --- Build system matrices ---
omega = 2 * pi * state.f0 / fs;
A = state.rho * [cos(omega), -sin(omega); sin(omega), cos(omega)];   % [2x2]
C = [1, 0];                                                           % [1x2]
Q = (state.sigma_w ^ 2) * eye(2);                                    % process noise
R = state.sigma_v ^ 2;                                                % observation noise

% --- Kalman filter: process each sample in chunk ---
phi_history = zeros(1, N);

for n = 1:N
    y_n = chunk(n);

    % Predict
    x_pred = A * state.x_hat;
    P_pred = A * state.P * A' + Q;

    % Update
    S  = C * P_pred * C' + R;
    K  = P_pred * C' / S;
    state.x_hat = x_pred + K * (y_n - C * x_pred);
    state.P     = (eye(2) - K * C) * P_pred;

    state.n_obs = state.n_obs + 1;

    % Wait 2*fs samples before reporting (transient convergence period)
    if state.n_obs < 2 * fs
        phi_history(n) = 0;
    else
        state.initialized = true;
    end

    % Current phase from posterior state
    phi_now = atan2(state.x_hat(2), state.x_hat(1));

    % Phase at t + shift: extrapolate using known oscillation frequency
    phi_future = phi_now + 2 * pi * state.f0 * (shift_ms / 1000);
    phi_future = atan2(sin(phi_future), cos(phi_future));  % wrap

    phi_history(n) = phi_future;
end

% --- Assemble output: [2 x N] ---
cos_hist = cos(phi_history);
sin_hist = sin(phi_history);
Pred = {[cos_hist; sin_hist]};

end
