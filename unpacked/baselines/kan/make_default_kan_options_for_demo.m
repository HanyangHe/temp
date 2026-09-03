function kanOpts=make_default_kan_options_for_demo(seed)
%MAKE_DEFAULT_KAN_OPTIONS_FOR_DEMO Pruned-KAN Feynman sweep defaults.
    if nargin<1||isempty(seed);seed=1;end
    kanOpts=kan_default_options(); kanOpts.seed=seed;
end
