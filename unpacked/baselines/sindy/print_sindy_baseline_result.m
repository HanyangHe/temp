function print_sindy_baseline_result(resultSindy)
%PRINT_SINDY_BASELINE_RESULT Print compact SINDy baseline metrics.

	fprintf('\n========================================\n');
	variant = lower(strtrim(char(getfield_default_local(resultSindy,'baselineVariant','standard_sindy'))));
	if strcmp(variant,'neural_sindy')
		fprintf('Neural-SINDy baseline: fixed neural-ridge replacement dictionary\n');
	else
		fprintf('Single-layer SINDy baseline with independent flat dictionary\n');
	end
	fprintf('========================================\n');
	fprintf('Dictionary source                       : %s\n', resultSindy.dictionarySource);
	if strcmp(variant,'neural_sindy') && isfield(resultSindy,'arch') && ...
			isfield(resultSindy.arch,'sindyDictionaryReport')
		report = resultSindy.arch.sindyDictionaryReport;
		fprintf('Standalone polynomial terms             : replaced by fixed neural-ridge bases\n');
		fprintf('Declared neural-ridge bases              : %d\n', ...
			getfield_default_local(report,'neuralCount',NaN));
		fprintf('Evaluated neural-ridge columns           : %d\n', ...
			getfield_default_local(resultSindy,'nNeuralTerms',NaN));
	end
	fprintf('Operator mode                           : %s\n', resultSindy.operatorMode);
	if isfield(resultSindy, 'solver')
		fprintf('Solver                                  : %s\n', resultSindy.solver);
	end
	if isfield(resultSindy, 'ridgeLambda')
		fprintf('Ridge lambda in LSQ solve               : %.3e\n', resultSindy.ridgeLambda);
	end
	if isfield(resultSindy, 'trainTime') && isfinite(resultSindy.trainTime)
		fprintf('Training wall time                      : %.3f s\n', resultSindy.trainTime);
	end
	if isfield(resultSindy, 'timeStats')
		fprintf('Dictionary / STLSQ / eval time          : %.3f / %.3f / %.3f s\n', ...
			getfield_default_local(resultSindy.timeStats, 'dictionaryTime', NaN), ...
			getfield_default_local(resultSindy.timeStats, 'stlsqTime', NaN), ...
			getfield_default_local(resultSindy.timeStats, 'evaluationTime', NaN));
	end
	if isfield(resultSindy, 'dictionarySupport') && isstruct(resultSindy.dictionarySupport)
		if getfield_default_local(resultSindy.dictionarySupport, 'applied', false)
			fprintf('PhDN compact support mapped to SINDy     : 1 (%d / %d rows before filtering)\n', ...
				getfield_default_local(resultSindy.dictionarySupport, 'nLibrarySupported', NaN), ...
				getfield_default_local(resultSindy.dictionarySupport, 'nLibraryTotal', resultSindy.nLibraryTotal));
		else
			fprintf('PhDN compact support mapped to SINDy     : 0 (%s)\n', ...
				getfield_default_local(resultSindy.dictionarySupport, 'reason', 'not available'));
		end
	end
	if isfield(resultSindy, 'stage0InitialGuessUnion') && ...
			isstruct(resultSindy.stage0InitialGuessUnion) && ...
			getfield_default_local(resultSindy.stage0InitialGuessUnion, 'enabled', false)
		u = resultSindy.stage0InitialGuessUnion;
		fprintf('Stage-0 SR guesses unioned into SINDy    : %d requested, %d added, %d duplicate\n', ...
			getfield_default_local(u,'nRequested',0), ...
			getfield_default_local(u,'nAdded',0), ...
			getfield_default_local(u,'nDuplicate',0));
	end
	fprintf('Library rows total / used               : %d / %d\n', resultSindy.nLibraryTotal, resultSindy.nLibraryUsed);
	if isfield(resultSindy,'libraryNumericalRankRaw')
		fprintf('Raw training-library numerical rank      : %d / %d\n', ...
			resultSindy.libraryNumericalRankRaw,resultSindy.nLibraryTotal);
	end
	fprintf('Invalid train/val/test/OOD rows         : %d / %d / %d / %d\n', ...
		resultSindy.nInvalidTrainRows, resultSindy.nInvalidValRows, ...
		resultSindy.nInvalidTestRows, resultSindy.nInvalidOodRows);
	fprintf('Selected STLSQ threshold                : %.6e\n', resultSindy.threshold);
	fprintf('Active terms / coefficients             : %d / %d\n', ...
		resultSindy.nActiveTerms, resultSindy.nActiveCoefficients);
	fprintf('Train MSE/RMSE                          : %.6e / %.6e\n', ...
		resultSindy.trainMetrics.mse, resultSindy.trainMetrics.rmse);
	fprintf('Validation MSE/RMSE                     : %.6e / %.6e\n', ...
		resultSindy.valMetrics.mse, resultSindy.valMetrics.rmse);
	fprintf('In-distribution test MSE/RMSE           : %.6e / %.6e\n', ...
		resultSindy.testMetrics.mse, resultSindy.testMetrics.rmse);
	if isfield(resultSindy, 'oodMetrics') && isfield(resultSindy.oodMetrics, 'mse') && isfinite(resultSindy.oodMetrics.mse)
		fprintf('OOD test MSE/RMSE                       : %.6e / %.6e\n', ...
			resultSindy.oodMetrics.mse, resultSindy.oodMetrics.rmse);
	end

	maxTerms = 30;
	if isfield(resultSindy, 'opts') && isfield(resultSindy.opts, 'maxTermsToPrint')
		maxTerms = resultSindy.opts.maxTermsToPrint;
	end
	if isfield(resultSindy, 'terms') && ~isempty(resultSindy.terms)
		coefMag = max(abs(resultSindy.Xi), [], 2);
		activeRows = find(coefMag > 0);
		[~, ord] = sort(coefMag(activeRows), 'descend');
		activeRows = activeRows(ord);
		nPrint = min(maxTerms, numel(activeRows));
		if nPrint > 0
			fprintf('Top active SINDy terms:\n');
			for k = 1:nPrint
				ir = activeRows(k);
				fprintf('  %4d  %-40s  coef=%+.6e\n', ...
					resultSindy.terms(ir).index, resultSindy.terms(ir).name, resultSindy.Xi(ir, 1));
			end
			if nPrint < numel(activeRows)
				fprintf('  ... %d more active terms not printed.\n', numel(activeRows) - nPrint);
			end
		end
	end
	fprintf('========================================\n');
end


function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
