function eqlOpts = make_default_eql_options_for_demo(seed)
%MAKE_DEFAULT_EQL_OPTIONS_FOR_DEMO Return unified EQL-Div sweep options.
    if nargin < 1 || isempty(seed); seed = 1; end
    eqlOpts = eql_default_options();
    eqlOpts.seed = seed;
end
