function s = output_denorm_derivative(normOpt, ny)
%OUTPUT_DENORM_DERIVATIVE dY_raw/dY_normalized for output postprocessing.
	if nargin < 2 || isempty(ny)
		ny = numel(getfield_default_norm_apply_local(normOpt, 'muY', []));
	end
	s = ones(1, ny);
	if nargin < 1 || isempty(normOpt) || ~isfield(normOpt, 'useOutputNorm') || ~normOpt.useOutputNorm
		return;
	end
	style = lower(strtrim(getfield_default_norm_apply_local(normOpt, 'style', 'mapminmax')));
	switch style
		case {'none','off'}
			s = ones(1, ny);
		case {'mapminmax','minmax'}
			s = 1 ./ normOpt.yGain;
			if isfield(normOpt, 'yIsConstant') && any(normOpt.yIsConstant)
				s(normOpt.yIsConstant) = 0;
			end
		case {'zscore','mapstd','standard'}
			s = normOpt.sigY;
		otherwise
			error('Unknown normalization style: %s.', style);
	end
	s(~isfinite(s)) = 1;
end

function val = getfield_default_norm_apply_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
