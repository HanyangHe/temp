function Yn = apply_output_norm(Y, normOpt)
%APPLY_OUTPUT_NORM Apply training-set output/target normalization.
	if nargin < 2 || isempty(normOpt)
		normOpt = default_norm_options();
	end
	Yn = Y;
	if ~isfield(normOpt, 'useOutputNorm') || ~normOpt.useOutputNorm
		return;
	end
	style = lower(strtrim(getfield_default_norm_apply_local(normOpt, 'style', 'mapminmax')));
	switch style
		case {'none','off'}
			return;
		case {'mapminmax','minmax'}
			Yn = (Y - normOpt.yOffset) .* normOpt.yGain + normOpt.ymin;
			if isfield(normOpt, 'yIsConstant') && any(normOpt.yIsConstant)
				Yn(:, normOpt.yIsConstant) = 0;
			end
		case {'zscore','mapstd','standard'}
			Yn = (Y - normOpt.muY) ./ normOpt.sigY;
		otherwise
			error('Unknown normalization style: %s.', style);
	end
	Yn(~isfinite(Yn)) = 0;
end

function val = getfield_default_norm_apply_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
