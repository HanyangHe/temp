function opts = make_default_sindy_options_for_demo()
%MAKE_DEFAULT_SINDY_OPTIONS_FOR_DEMO Default independent SINDy baseline options.
%
% In v69, SINDy is not allowed to depend on PhDN/SR masks by default.  It uses
% a broad flat dictionary generated from raw inputs: polynomial monomials under
% a chosen total degree plus traditional unary-operator and cross terms.
	opts = sindy_default_options();
end
