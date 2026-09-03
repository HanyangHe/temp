function mlpOpts = mlp_default_options()
%MLP_DEFAULT_OPTIONS Defaults for the KAN-paper-style MLP sweep baseline.
%
% The default protocol borrows the Feynman architecture sweep from the KAN
% paper: fixed width 20, network depths 2:6, and Tanh/ReLU/SiLU activations.
% In pykan terminology depth is the number of affine layers, so a depth-D MLP
% has D-1 hidden layers. Each outer framework round supplies one random seed;
% validation MSE selects one of the 15 candidates within that round.

    mlpDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(mlpDir));
    mlpOpts = struct();
    mlpOpts.protocol = 'kan_feynman_sweep'; % kan_feynman_sweep | fixed_fitnet
    mlpOpts.seed = 1;

    % KAN-paper Feynman MLP structure sweep.
    mlpOpts.width = 20;
    mlpOpts.depthList = 2:6;
    mlpOpts.activationList = {'tanh','relu','silu'};
    mlpOpts.depthConvention = 'number_of_affine_layers';
    mlpOpts.selectionMetric = 'validation_mse';

    % Official-pykan-style continuous training.
    mlpOpts.optimizer = 'LBFGS';
    mlpOpts.steps = 500;
    mlpOpts.learningRate = 1.0;
    mlpOpts.dtype = 'float64';
    mlpOpts.device = 'cpu';
    mlpOpts.torchNumThreads = 0; % 0 leaves the PyTorch default unchanged.
    mlpOpts.normalizeInputs = true;
    mlpOpts.normalizeOutputs = true;
    mlpOpts.pythonExe = 'python';
    mlpOpts.pykanRoot = fullfile(projectRoot, 'third_party', 'pykan');
    mlpOpts.workRoot = fullfile(projectRoot, 'tmp', 'kan_paper_mlp_runs');
    mlpOpts.displaySweepTable = true;
    mlpOpts.verbose = true;

    % Preserved legacy fixed-fitnet protocol.
    mlpOpts.hiddenLayerSizes = [64,64];
    mlpOpts.trainFcn = 'trainlm';
    mlpOpts.performFcn = 'mse';
    mlpOpts.hiddenTransferFcn = 'tansig';
    mlpOpts.outputTransferFcn = 'purelin';
    mlpOpts.inputProcessFcns = {'removeconstantrows','mapminmax'};
    mlpOpts.outputProcessFcns = {'removeconstantrows','mapminmax'};
    mlpOpts.trainParam.epochs = 500;
    mlpOpts.trainParam.max_fail = 30;
    mlpOpts.trainParam.min_grad = 1e-10;
    mlpOpts.trainParam.showWindow = false;
    mlpOpts.trainParam.showCommandLine = false;
end
