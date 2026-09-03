function s = format_variable_domain(domain, precision)
%FORMAT_VARIABLE_DOMAIN Convert a per-variable interval-union domain to text.

	if nargin < 2 || isempty(precision)
		precision = '%.4g';
	end

	domain = normalize_task_domain(domain);
	parts = cell(1, numel(domain.intervals));
	for j = 1:numel(domain.intervals)
		if isfield(domain, 'variableNames') && numel(domain.variableNames) >= j
			name = domain.variableNames{j};
		else
			name = sprintf('x%d', j);
		end
		I = domain.intervals{j};
		pieces = cell(1, size(I, 1));
		for k = 1:size(I, 1)
			pieces{k} = sprintf(['[', precision, ', ', precision, ']'], I(k, 1), I(k, 2));
		end
		parts{j} = sprintf('%s: %s', name, strjoin(pieces, ' U '));
	end
	s = strjoin(parts, '; ');
end
