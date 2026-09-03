function Xn = apply_input_norm(X, normOpt)
%APPLY_INPUT_NORM Apply training-set input normalization to new inputs.
	if nargin < 2 || isempty(normOpt)
		normOpt = default_norm_options();
	end
	Xn = X;
	if ~isfield(normOpt, 'useInputNorm') || ~normOpt.useInputNorm
		return;
	end
	style = lower(strtrim(getfield_default_norm_apply_local(normOpt, 'style', 'mapminmax')));
	switch style
		case {'none','off'}
			return;
		case {'mapminmax','minmax'}
			Xn = (X - normOpt.xOffset) .* normOpt.xGain + normOpt.ymin;
			if isfield(normOpt, 'xIsConstant') && any(normOpt.xIsConstant)
				Xn(:, normOpt.xIsConstant) = 0;
			end
		case {'zscore','mapstd','standard'}
			Xn = (X - normOpt.muX) ./ normOpt.sigX;
		otherwise
			error('Unknown normalization style: %s.', style);
	end
	Xn(~isfinite(Xn)) = 0;
end

function val = getfield_default_norm_apply_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
