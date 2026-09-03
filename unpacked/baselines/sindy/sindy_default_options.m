function opts = sindy_default_options()
%SINDY_DEFAULT_OPTIONS Default options for the independent single-layer SINDy baseline.
%
% By default, this baseline builds its own broad/general flat dictionary from
% raw inputs.  It does not reuse PhDN masks, SR-derived DAGs, or compact case
% dictionaries.  The library contains polynomial monomials up to a given total
% degree, traditional unary operator terms, and optional variable-operator
% cross terms.
%
% Legacy mode can still use the same one-branch dictionary Phi(x) as PhDN, but it is
% always evaluated with true physical operators.  It solves
%
%   Y ~= Phi_true(X) * Xi
%
% with the original SINDy-style sequential thresholded least-squares (STLSQ)
% algorithm.  By default, each least-squares solve uses MATLAB backslash, as
% in the original sparsifyDynamics.m implementation.

	opts = struct();

	% Sequential thresholded least-squares thresholds.  The first entry zero
	% corresponds to dense least squares with no thresholding.
	opts.thresholdList = [0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3];

	% Maximum number of sequential threshold/refit passes for each threshold.
	% This matches the original SINDy reference implementation default.
	opts.maxSTLSQIter = 10;

	% Default: original SINDy least-squares solve, Theta \ Y.  Set this to a
	% positive value only if a numerically regularized variant is intentionally
	% desired.
	opts.ridgeLambda = 0;

	% Validation model selection.  A small complexity tie-breaker is useful when
	% validation errors are almost identical.
	opts.acceptRelTol = 1e-10;
	opts.complexityTieWeight = 0;

	% Dictionary processing.
	opts.removeInvalidTrainRows = true;
	opts.removeNearConstantRows = false;
	opts.nearConstantStdTol = 1e-12;
	opts.centerScaleLibrary = false;
	opts.libraryScaleFloor = 1e-12;

	% Independent baseline dictionary.
	% 'general'  : broad flat SINDy dictionary generated from raw inputs.
	% 'phdn_phi' : legacy mode using the current PhDN one-branch Phi(x).
	opts.dictionaryMode = 'general';
	opts.includePolynomialTerms = true;
	opts.polyOrder = 2;
	opts.unaryOperators = {'inv','sqrt','exp','sin','cos','log'};
	opts.includeUnaryOnMonomials = true;
	opts.includeOperatorCrossTerms = true;
	opts.includeSinCosPair = false;

	% By default the historical system-identification runner mirrors Stage-0
	% initial guesses into SINDy. Case-specific matched-library comparisons may
	% disable this so the declared dictionary dimension remains exact.
	opts.syncStage0InitialGuesses = true;
	opts.strictLibraryAssertions = false;
	opts.expectedLibrarySize = [];
	opts.expectedNeuralCount = [];
	opts.maxLibraryTerms = Inf;

	% Fixed neural-ridge replacement block used by dictionaryMode='neural_general'.
	% These defaults mirror the Lorenz--96 PhDN Stage-1 augmentation.
	opts.neuralCount = [];
	opts.neuralActivation = 'tanh';
	opts.neuralQuantiles = [0.25,0.50,0.75];
	opts.neuralScales = [0.5,1,2];
	opts.neuralPoolRatio = 3;
	opts.neuralSeed = 11886;
	opts.neuralStdFloor = 1e-10;
	opts.neuralVarianceThreshold = 1e-8;
	opts.neuralCorrelationThreshold = 0.995;
	opts.neuralEnsureFullDirectionalSpan = true;

	% Internal fair-prior channel. System-identification runners automatically
	% populate these fields from opts.stage0.pysr.initialGuesses whenever that
	% shared SR seed library is enabled. Users normally should not edit them.
	opts.stage0InitialGuessTerms = {};
	opts.stage0InitialGuessSyncInfo = struct();

	% PhDN compact support is disabled by default because SINDy is an independent
	% baseline in v69.  This field is honored only in dictionaryMode='phdn_phi'.
	opts.usePhdnDictionarySupport = false;

	% Reporting.
	opts.verbose = true;
	opts.maxTermsToPrint = 30;
end
