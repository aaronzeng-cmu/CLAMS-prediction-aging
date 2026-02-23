function predictor = LoadAgingPredictor(model_path, model_type, varargin)
% LoadAgingPredictor  Load an exported aging SO phase predictor from disk.
%
% Supports two model types:
%   'lstm' — Reconstructs a dlnetwork from PyTorch-exported .mat weights.
%            Gate ordering: PyTorch [i|f|g|o] == MATLAB [i|f|g|o] — no reordering.
%            Bias: saved as combined (bias_ih + bias_hh) in (4H,1) column.
%   'tcn'  — Loads ONNX via importNetworkFromONNX (requires MATLAB R2023b+).
%            Uses a fixed circular buffer of size predictor.rf samples.
%
% Usage:
%   predictor = LoadAgingPredictor('models/lstm_shift_100ms.mat', 'lstm')
%   predictor = LoadAgingPredictor('models/tcn_shift_100ms.onnx', 'tcn')
%   predictor = LoadAgingPredictor(..., 'shift_ms', 100)
%
% Output struct fields:
%   .net         — dlnetwork (lstm) or DAGNetwork (tcn)
%   .model_type  — 'lstm' or 'tcn'
%   .shift_ms    — shift value (ms); parsed from filename if not given
%   .buffer      — [C x rf] circular feature buffer (tcn only)
%   .rf          — receptive field length (tcn only)

p = inputParser;
addRequired(p, 'model_path', @ischar);
addRequired(p, 'model_type', @ischar);
addParameter(p, 'shift_ms', [], @isnumeric);
parse(p, model_path, model_type, varargin{:});

predictor.model_type = lower(model_type);
predictor.shift_ms   = p.Results.shift_ms;

% Parse shift_ms from filename if not provided
if isempty(predictor.shift_ms)
    tok = regexp(model_path, 'shift_(\d+)ms', 'tokens');
    if ~isempty(tok)
        predictor.shift_ms = str2double(tok{1}{1});
    else
        predictor.shift_ms = 0;
    end
end

switch predictor.model_type

    % ------------------------------------------------------------------
    case 'lstm'
    % ------------------------------------------------------------------
        w = load(model_path);
        C          = double(w.input_size(1));
        H          = double(w.hidden_size(1));
        num_layers = double(w.num_layers(1));

        % Build layer array
        % Gate ordering: PyTorch [i|f|g|o] rows = MATLAB lstmLayer [i;f;g;o] rows
        % No reordering needed — bias already combined (bias_ih + bias_hh) at export
        layers = sequenceInputLayer(C, 'Name', 'input');

        for k = 1:num_layers
            W_ih  = double(w.(sprintf('W_ih_%d', k)));   % (4H x C_in)
            W_hh  = double(w.(sprintf('W_hh_%d', k)));   % (4H x H)
            bias  = double(w.(sprintf('bias_%d', k)));   % (4H x 1)

            layers = [layers; ...
                lstmLayer(H, ...
                    'InputWeights',     W_ih, ...
                    'RecurrentWeights', W_hh, ...
                    'Bias',             bias, ...
                    'Name',             sprintf('lstm_%d', k))];
        end

        W_fc = double(w.W_fc);          % (2 x H)
        b_fc = double(w.b_fc);          % (2 x 1)
        layers = [layers; ...
            fullyConnectedLayer(2, ...
                'Weights', W_fc, ...
                'Bias',    b_fc, ...
                'Name',    'fc')];

        net = dlnetwork(layers);
        predictor.net = resetState(net);

    % ------------------------------------------------------------------
    case 'tcn'
    % ------------------------------------------------------------------
        if ~exist('importNetworkFromONNX', 'file')
            error('LoadAgingPredictor:onnxRequired', ...
                'importNetworkFromONNX requires MATLAB R2023b or later.');
        end
        net = importNetworkFromONNX(model_path, 'TargetNetwork', 'dlnetwork');
        predictor.net = net;

        % Infer receptive field from ONNX input shape (dim 3 = seq_len)
        in_info = net.InputNames;
        % Default to 512 if cannot determine
        predictor.rf = 512;
        try
            in_layer = net.Layers(1);
            predictor.rf = in_layer.InputSize(end);
        catch
        end

        % Circular buffer initialized to zeros: [C x rf]
        C = 3;
        predictor.buffer = zeros(C, predictor.rf, 'single');

    otherwise
        error('LoadAgingPredictor:unknownType', ...
            'Unknown model_type ''%s''. Use ''lstm'' or ''tcn''.', model_type);
end

end
