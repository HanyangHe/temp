function trainMask = build_training_mask(Coef_template, opts)
%BUILD_TRAINING_MASK Build the coefficient mask optimized by GA/LSQ.
%
% Strong-prior case:
%   opts.training.useAdmissibleMask = true and opts.training.admissibleA
%   is used directly.  Thus GA/LSQ optimizes the full prior-specified support.
%
% previous-version dictionary-support mode:
%   opts.training.dictionarySupportA optionally supplies an additional
%   dictionary-level support mask.  It is intersected with either the exact
%   admissible mask or the default all-allowed mask.  For weak-prior Feynman
%   cases this is used to keep only the basis functions required by the
%   corresponding strong-prior construction, without imposing the exact
%   row-wise strong-prior support.
%
% No-mask case:
%   all coefficients are allowed unless dictionarySupportA is supplied.

	useAdmissible = isfield(opts, 'training') && isfield(opts.training, 'useAdmissibleMask') && ...
			logical(opts.training.useAdmissibleMask) && ...
			isfield(opts.training, 'admissibleA') && ~isempty(opts.training.admissibleA);

	if useAdmissible
		trainMask = opts.training.admissibleA;
		check_mask_size_local(Coef_template, trainMask, 'Admissible mask');
	else
		defaultAllowed = true;
		if isfield(opts, 'training') && isfield(opts.training, 'admissibleDefaultAllowed') && ...
				~isempty(opts.training.admissibleDefaultAllowed)
			defaultAllowed = logical(opts.training.admissibleDefaultAllowed);
		end
		trainMask = make_admissible_mask_like(Coef_template, defaultAllowed);
	end

	% previous-version optional task/dictionary-level support restriction.
	if isfield(opts, 'training') && isfield(opts.training, 'dictionarySupportA') && ...
			~isempty(opts.training.dictionarySupportA)
		supportA = opts.training.dictionarySupportA;
		check_mask_size_local(Coef_template, supportA, 'Dictionary-support mask');
		trainMask = intersect_masks_local(trainMask, supportA);
	end
end

function out = intersect_masks_local(a, b)
	out = a;
	for j = 1:size(a, 2)
		for i = 1:j
			if isempty(a{i, j})
				continue;
			end
			out{i, j} = logical(a{i, j}) & logical(b{i, j});
		end
	end
end

function check_mask_size_local(Coef_template, mask, label)
	if nargin < 3 || isempty(label)
		label = 'Mask';
	end
	for j = 1:size(Coef_template, 2)
		for i = 1:j
			if isempty(Coef_template{i, j})
				continue;
			end
			if i > size(mask, 1) || j > size(mask, 2) || isempty(mask{i, j})
				error('%s is missing block {%d,%d}.', label, i, j);
			end
			if ~isequal(size(Coef_template{i, j}), size(mask{i, j}))
				error('%s size mismatch at block {%d,%d}.', label, i, j);
			end
		end
	end
end
