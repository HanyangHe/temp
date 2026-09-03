function result = attach_data_split_to_result(result, XTrain, YTrain, XVal, YVal, XTest, YTest, XOod, YOod, task, opts, oodDomain)
%ATTACH_DATA_SPLIT_TO_RESULT Save the exact PhDN data split into result.
%
% This helper should be called inside phdnn_identify.m after the exact
% train/validation/test/OOD arrays have been generated.
%
% Required:
%   result = attach_data_split_to_result(result, XTrain, YTrain, XVal, YVal,
%       XTest, YTest, XOod, YOod, task, opts, oodDomain)
%
% If opts.output.saveDataSplit is false or missing, result is returned
% unchanged.

	if nargin < 12
		oodDomain = [];
	end

	if ~should_save_data_split_local(opts)
		return;
	end

	result.dataSplit = struct();

	result.dataSplit.XTrain = XTrain;
	result.dataSplit.YTrain = YTrain;

	result.dataSplit.XVal = XVal;
	result.dataSplit.YVal = YVal;

	result.dataSplit.XTest = XTest;
	result.dataSplit.YTest = YTest;

	if nargin >= 8 && ~isempty(XOod) && ~isempty(YOod)
		result.dataSplit.XOod = XOod;
		result.dataSplit.YOod = YOod;
	else
		result.dataSplit.XOod = [];
		result.dataSplit.YOod = [];
	end

	if nargin >= 10 && ~isempty(task) && isstruct(task) && isfield(task, 'domain')
		result.dataSplit.domain = task.domain;
	end

	if nargin >= 12 && ~isempty(oodDomain)
		result.dataSplit.oodDomain = oodDomain;
	elseif nargin >= 10 && ~isempty(task) && isstruct(task) && isfield(task, 'oodDomain')
		result.dataSplit.oodDomain = task.oodDomain;
	else
		result.dataSplit.oodDomain = [];
	end

	result.dataSplit.note = ['Exact train/validation/test/OOD data used by ', ...
		'PhDN, saved for strict external baseline comparison.'];
end

function tf = should_save_data_split_local(opts)
	tf = false;

	if isstruct(opts) && isfield(opts, 'output') && isstruct(opts.output) && ...
			isfield(opts.output, 'saveDataSplit') && ~isempty(opts.output.saveDataSplit)
		tf = logical(opts.output.saveDataSplit);
	end
end
