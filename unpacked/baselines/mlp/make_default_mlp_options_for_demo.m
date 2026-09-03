function mlpOpts = make_default_mlp_options_for_demo(seed)
%MAKE_DEFAULT_MLP_OPTIONS_FOR_DEMO KAN-paper-style MLP sweep defaults.
    if nargin < 1 || isempty(seed); seed = 1; end
    mlpOpts = mlp_default_options();
    mlpOpts.seed = seed;
end
