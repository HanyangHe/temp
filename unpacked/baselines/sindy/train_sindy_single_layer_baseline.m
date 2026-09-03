function result = train_sindy_single_layer_baseline(Xtr, Ytr, Xval, Yval, Xte, Yte, XOod, YOod, arch, opts)
%TRAIN_SINDY_SINGLE_LAYER_BASELINE Single-layer SINDy using a flat true-operator Phi(x).
%
% Model:
%   Yhat = Phi_true(X) * Xi
%
% where Phi_true is a one-layer explicit dictionary evaluated on raw inputs.
% In v69 default usage, this dictionary is generated independently by
% make_sindy_general_arch.m rather than inherited from PhDN.
% Sparse coefficients are obtained by the original SINDy-style sequential
% thresholded least-squares (STLSQ) procedure:
%
%   Xi = Theta \ Y;
%   for iter = 1:maxSTLSQIter
%       smallinds = abs(Xi) < lambda;
%       Xi(smallinds) = 0;
%       for each output dimension
%           refit the remaining columns by least squares
%       end
%   end
%
% The threshold lambda is selected by validation MSE.

	if nargin < 10 || isempty(opts)
		opts = sindy_default_options();
	end
	if nargin < 8
		XOod = [];
		YOod = [];
	end

	tTotal = tic;

	% SINDy baseline is intentionally true-operator, even if the main PhDN is a
	% surrogate-operator model.
	arch.operatorMode = 'true';

	tDictionary = tic;
	[PhiTrRaw, termInfo, invalidTrainRows] = build_sindy_phi_local(Xtr, arch);
	[PhiValRaw, ~, invalidValRows] = build_sindy_phi_local(Xval, arch);
	[PhiTeRaw, ~, invalidTestRows] = build_sindy_phi_local(Xte, arch);

	if ~isempty(XOod) && ~isempty(YOod)
		[PhiOodRaw, ~, invalidOodRows] = build_sindy_phi_local(XOod, arch);
	else
		PhiOodRaw = [];
		invalidOodRows = false(size(PhiTrRaw, 2), 1);
	end
	dictionaryTime = toc(tDictionary);

	expectedLibrarySize = getfield_default_local(opts,'expectedLibrarySize',[]);
	if logical(getfield_default_local(opts,'strictLibraryAssertions',false)) && ...
			~isempty(expectedLibrarySize) && size(PhiTrRaw,2) ~= expectedLibrarySize
		error('SINDy evaluated library size mismatch: got %d columns, expected %d.', ...
			size(PhiTrRaw,2),expectedLibrarySize);
	end
	termNamesAll = {termInfo.name};
	nNeuralTerms = nnz(startsWith(termNamesAll,'tanh('));
	expectedNeuralCount = getfield_default_local(opts,'expectedNeuralCount',[]);
	if logical(getfield_default_local(opts,'strictLibraryAssertions',false)) && ...
			~isempty(expectedNeuralCount) && nNeuralTerms ~= expectedNeuralCount
		error('SINDy evaluated neural basis count mismatch: got %d, expected %d.', ...
			nNeuralTerms,expectedNeuralCount);
	end
	libraryNumericalRankRaw = rank(PhiTrRaw);

	% Do not silently discard an SR-seed prior term from the SINDy comparator.
	% The union builder records the exact row used for every requested guess.
	stage0GuessUnion = get_stage0_guess_union_local(arch);
	if stage0GuessUnion.enabled
		idx = unique(stage0GuessUnion.libraryIndices(:));
		if any(idx < 1) || any(idx > size(PhiTrRaw,2))
			error('SINDy Stage-0 guess-union indices are inconsistent with the dictionary size.');
		end
		bad = idx(invalidTrainRows(idx));
		if ~isempty(bad)
			names = {termInfo(bad).name};
			error(['A Stage0SRInitialGuesses expression could not be evaluated by the ', ...
				'SINDy dictionary and would violate prior fairness: %s'], strjoin(names, ', '));
		end
	end

	rowKeep = true(size(PhiTrRaw, 2), 1);
	[supportKeep, supportInfo] = resolve_sindy_library_support_local(arch, opts, size(PhiTrRaw, 2));
	rowKeep = rowKeep & supportKeep(:);
	if getfield_default_local(opts, 'removeInvalidTrainRows', true)
		rowKeep = rowKeep & ~invalidTrainRows(:);
	end
	if getfield_default_local(opts, 'removeNearConstantRows', false)
		stdTol = getfield_default_local(opts, 'nearConstantStdTol', 1e-12);
		rowKeep = rowKeep & (std(PhiTrRaw, 0, 1).' > stdTol | abs(mean(PhiTrRaw, 1).') > stdTol);
	end

	PhiTr = PhiTrRaw(:, rowKeep);
	PhiVal = PhiValRaw(:, rowKeep);
	PhiTe = PhiTeRaw(:, rowKeep);
	if ~isempty(PhiOodRaw)
		PhiOod = PhiOodRaw(:, rowKeep);
	else
		PhiOod = [];
	end
	termsKept = termInfo(rowKeep);

	if isempty(PhiTr)
		error('SINDy baseline has no valid dictionary rows after filtering.');
	end

	libScale = ones(1, size(PhiTr, 2));
	libMean = zeros(1, size(PhiTr, 2));
	if getfield_default_local(opts, 'centerScaleLibrary', false)
		libMean = mean(PhiTr, 1);
		libScale = std(PhiTr, 0, 1);
		libScale(~isfinite(libScale) | libScale < opts.libraryScaleFloor) = 1;
		PhiTrFit = (PhiTr - libMean) ./ libScale;
		PhiValFit = (PhiVal - libMean) ./ libScale;
	else
		PhiTrFit = PhiTr;
		PhiValFit = PhiVal;
	end

	thresholdList = getfield_default_local(opts, 'thresholdList', 0);
	thresholdList = unique(thresholdList(:).');
	if isempty(thresholdList)
		thresholdList = 0;
	end

	best = struct();
	best.valMSE = Inf;
	best.score = Inf;
	best.threshold = NaN;
	best.XiFit = [];
	best.activeMask = [];
	best.trainMetrics = [];
	best.valMetrics = [];

	tSTLSQ = tic;
	for k = 1:numel(thresholdList)
		thr = thresholdList(k);
		[XiFit, activeMask] = stlsq_fit_local(PhiTrFit, Ytr, thr, opts);
		YtrPred = PhiTrFit * XiFit;
		YvalPred = PhiValFit * XiFit;
		trainMetrics = compute_regression_metrics(YtrPred, Ytr);
		valMetrics = compute_regression_metrics(YvalPred, Yval);
		valMSE = valMetrics.mse;
		if ~isfinite(valMSE)
			valMSE = Inf;
		end
		nActive = nnz(abs(XiFit) > 0);
		score = valMSE + getfield_default_local(opts, 'complexityTieWeight', 0) * nActive;
		if score < best.score * (1 - getfield_default_local(opts, 'acceptRelTol', 1e-10)) || ...
				(abs(score - best.score) <= getfield_default_local(opts, 'acceptRelTol', 1e-10) * max(1, abs(best.score)) && nActive < nnz(abs(best.XiFit) > 0))
			best.valMSE = valMSE;
			best.score = score;
			best.threshold = thr;
			best.XiFit = XiFit;
			best.activeMask = activeMask;
			best.trainMetrics = trainMetrics;
			best.valMetrics = valMetrics;
		end
	end
	stlsqTime = toc(tSTLSQ);

	Xi = best.XiFit;
	if getfield_default_local(opts, 'centerScaleLibrary', false)
		% Convert scaled-library coefficients to raw-library coefficients.  This
		% keeps prediction code simple.  The intercept shift is folded into the
		% constant row if a constant row exists; otherwise raw prediction uses the
		% scaled features stored below.
		XiRaw = Xi ./ libScale.';
		constIdx = find(strcmp({termsKept.name}, '1'), 1);
		if ~isempty(constIdx)
			XiRaw(constIdx, :) = XiRaw(constIdx, :) - (libMean ./ libScale) * Xi;
			useScaledPrediction = false;
		else
			XiRaw = Xi;
			useScaledPrediction = true;
		end
	else
		XiRaw = Xi;
		useScaledPrediction = false;
	end

	if useScaledPrediction
		predictFcn = @(PhiRaw) ((PhiRaw - libMean) ./ libScale) * XiRaw;
	else
		predictFcn = @(PhiRaw) PhiRaw * XiRaw;
	end

	tEval = tic;
	YtrPred = predictFcn(PhiTr);
	YvalPred = predictFcn(PhiVal);
	YtePred = predictFcn(PhiTe);
	trainMetrics = compute_regression_metrics(YtrPred, Ytr);
	valMetrics = compute_regression_metrics(YvalPred, Yval);
	testMetrics = compute_regression_metrics(YtePred, Yte);

	if ~isempty(PhiOod)
		YoodPred = predictFcn(PhiOod);
		oodMetrics = compute_regression_metrics(YoodPred, YOod);
	else
		YoodPred = [];
		oodMetrics = empty_metrics_local();
	end
	evaluationTime = toc(tEval);
	trainTime = toc(tTotal);

	result = struct();
	result.method = 'independent_single_layer_sindy_general_dictionary_stlsq';
	result.solver = 'STLSQ_original_sindy_style';
	result.opts = opts;
	result.arch = arch;
	result.Xi = XiRaw;
	result.XiFit = best.XiFit;
	result.threshold = best.threshold;
	result.thresholdList = thresholdList;
	result.ridgeLambda = getfield_default_local(opts, 'ridgeLambda', 0);
	result.nLibraryTotal = size(PhiTrRaw, 2);
	result.nLibraryUsed = size(PhiTr, 2);
	result.libraryNumericalRankRaw = libraryNumericalRankRaw;
	result.nNeuralTerms = nNeuralTerms;
	result.expectedLibrarySize = expectedLibrarySize;
	result.expectedNeuralCount = expectedNeuralCount;
	result.nInvalidTrainRows = nnz(invalidTrainRows);
	result.nInvalidValRows = nnz(invalidValRows);
	result.nInvalidTestRows = nnz(invalidTestRows);
	result.nInvalidOodRows = nnz(invalidOodRows);
	result.rowKeep = rowKeep;
	result.dictionarySupport = supportInfo;
	result.stage0InitialGuessUnion = stage0GuessUnion;
	result.terms = termsKept;
	result.nActiveCoefficients = nnz(abs(XiRaw) > 0);
	result.nActiveTerms = nnz(any(abs(XiRaw) > 0, 2));
	result.trainMetrics = trainMetrics;
	result.valMetrics = valMetrics;
	result.testMetrics = testMetrics;
	result.oodMetrics = oodMetrics;
	result.trainTime = trainTime;
	result.timeStats = struct();
	result.timeStats.total = trainTime;
	result.timeStats.dictionaryTime = dictionaryTime;
	result.timeStats.stlsqTime = stlsqTime;
	result.timeStats.evaluationTime = evaluationTime;
	result.prediction = struct('Ytr', YtrPred, 'Yval', YvalPred, 'Yte', YtePred, 'Yood', YoodPred);
	result.libraryNormalization = struct('enabled', getfield_default_local(opts, 'centerScaleLibrary', false), ...
		'mean', libMean, 'scale', libScale, 'useScaledPrediction', useScaledPrediction);
end



function info = get_stage0_guess_union_local(arch)
	info = struct('enabled',false,'requestedTerms',{{}},'nRequested',0, ...
		'nAdded',0,'nDuplicate',0,'addedTerms',{{}},'duplicateTerms',{{}}, ...
		'libraryIndices',zeros(0,1),'libraryTerms',{{}});
	if isstruct(arch) && isfield(arch, 'sindyDictionaryReport') && ...
			isstruct(arch.sindyDictionaryReport) && ...
			isfield(arch.sindyDictionaryReport, 'stage0InitialGuessUnion') && ...
			~isempty(arch.sindyDictionaryReport.stage0InitialGuessUnion)
		info = arch.sindyDictionaryReport.stage0InitialGuessUnion;
	elseif isstruct(arch) && isfield(arch, 'caseDictionary') && ...
			isstruct(arch.caseDictionary) && ...
			isfield(arch.caseDictionary, 'stage0InitialGuessUnion') && ...
			~isempty(arch.caseDictionary.stage0InitialGuessUnion)
		info = arch.caseDictionary.stage0InitialGuessUnion;
	end
	if ~isfield(info,'enabled'); info.enabled = false; end
end

function [supportKeep, supportInfo] = resolve_sindy_library_support_local(arch, opts, nTerms)
%RESOLVE_SINDY_LIBRARY_SUPPORT_LOCAL Apply PhDN compact dictionary support to SINDy.
%
% The previous-version weak-prior PhDN route may restrict each coefficient block to a
% compact set of dictionary columns through opts.training.dictionarySupportA.
% A single-layer SINDy library only has one input Phi(x), so the fair mapping
% is the column union of the PhDN input block A{1,1}, provided its number of
% columns matches the SINDy Phi(x) library size.  If the mapping is absent or
% size-incompatible, the SINDy baseline falls back to the full library and
% records the reason in supportInfo.

	if nargin < 3 || isempty(nTerms)
		nTerms = 0;
	end
	supportKeep = true(nTerms, 1);
	supportInfo = struct();
	supportInfo.enabled = false;
	supportInfo.applied = false;
	supportInfo.source = 'full_sindy_phi';
	supportInfo.reason = 'PhDN dictionary support not requested or unavailable';
	supportInfo.nLibraryTotal = nTerms;
	supportInfo.nLibrarySupported = nTerms;
	supportInfo.keepRows = supportKeep;

	if ~getfield_default_local(opts, 'usePhdnDictionarySupport', true)
		supportInfo.enabled = false;
		supportInfo.reason = 'opts.usePhdnDictionarySupport=false';
		return;
	end
	supportInfo.enabled = true;

	if ~isstruct(arch) || ~isfield(arch, 'sindyLibrarySupport') || isempty(arch.sindyLibrarySupport)
		supportInfo.reason = 'arch.sindyLibrarySupport is missing';
		return;
	end

	candidate = arch.sindyLibrarySupport;
	if isstruct(candidate)
		if isfield(candidate, 'keepRows') && ~isempty(candidate.keepRows)
			supportKeepCandidate = logical(candidate.keepRows(:));
		else
			supportInfo.reason = 'arch.sindyLibrarySupport.keepRows is missing';
			return;
		end
		if isfield(candidate, 'source') && ~isempty(candidate.source)
			supportInfo.source = candidate.source;
		else
			supportInfo.source = 'phdn_dictionary_support_A11_column_union';
		end
	else
		supportKeepCandidate = logical(candidate(:));
		supportInfo.source = 'phdn_dictionary_support_A11_column_union';
	end

	if numel(supportKeepCandidate) ~= nTerms
		supportInfo.reason = sprintf('support length %d does not match SINDy library size %d', numel(supportKeepCandidate), nTerms);
		return;
	end
	if ~any(supportKeepCandidate)
		supportInfo.reason = 'mapped PhDN dictionary support is empty';
		return;
	end

	supportKeep = supportKeepCandidate(:);
	supportInfo.applied = true;
	supportInfo.reason = 'using PhDN effective compact dictionary support mapped from A{1,1}';
	supportInfo.nLibrarySupported = nnz(supportKeep);
	supportInfo.keepRows = supportKeep;
end

function [Phi, terms, invalidRows] = build_sindy_phi_local(X, arch)
	H = X.';
	branch = build_branch_cache(H, arch, 1, {});
	Phi = branch.Phi.';
	invalidRows = branch.PhiInvalidRows(:);
	terms = branch_dictionary_terms(size(X, 2), arch, 'x', 1);
	if numel(terms) ~= size(Phi, 2)
		% Keep robust metadata even if future dictionary options change.
		terms = struct('index', {}, 'name', {}, 'type', {}, 'opName', {}, 'isIdentityCancellation', {});
		for k = 1:size(Phi, 2)
			terms(k).index = k; %#ok<AGROW>
			terms(k).name = sprintf('phi%d', k); %#ok<AGROW>
			terms(k).type = 'unknown'; %#ok<AGROW>
			terms(k).opName = ''; %#ok<AGROW>
			terms(k).isIdentityCancellation = false; %#ok<AGROW>
		end
	end
end

function [Xi, activeMask] = stlsq_fit_local(Theta, Y, lambda, opts)
%STLSQ_FIT_LOCAL Original SINDy-style sequential thresholded LSQ.
%
% This intentionally follows sparsifyDynamics.m from Brunton et al.:
%   Xi = Theta \ Y;
%   smallinds = abs(Xi) < lambda;
%   Xi(smallinds) = 0;
%   refit each output using only the remaining large terms.

	maxIter = getfield_default_local(opts, 'maxSTLSQIter', 10);
	Xi = solve_lsq_local(Theta, Y, opts);

	if lambda <= 0
		activeMask = abs(Xi) > 0;
		return;
	end

	for iter = 1:maxIter
		smallinds = abs(Xi) < lambda;
		Xi(smallinds) = 0;

		for ind = 1:size(Y, 2)
			biginds = ~smallinds(:, ind);
			if any(biginds)
				Xi(biginds, ind) = solve_lsq_local(Theta(:, biginds), Y(:, ind), opts);
			end
			Xi(~biginds, ind) = 0;
		end
	end

	activeMask = abs(Xi) > 0;
end

function Xi = solve_lsq_local(A, B, opts)
%SOLVE_LSQ_LOCAL Least-squares solve for SINDy STLSQ.
%
% Default behavior is the original paper/code style, A \ B.  Optional ridge
% regularization is available only when opts.ridgeLambda > 0.

	lambda = getfield_default_local(opts, 'ridgeLambda', 0);
	if lambda > 0
		Xi = (A.' * A + lambda * eye(size(A, 2))) \ (A.' * B);
	else
		Xi = A \ B;
	end
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end

function metrics = empty_metrics_local()
	metrics = struct();
	metrics.mse = NaN;
	metrics.rmse = NaN;
	metrics.nrmse = NaN;
	metrics.mae = NaN;
	metrics.nmae = NaN;
	metrics.maxAbs = NaN;
end
