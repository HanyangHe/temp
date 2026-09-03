function [base, elapsedTime] = build_stage0_base_dictionary(task, archBase, data, options)
%BUILD_STAGE0_BASE_DICTIONARY Construct and cache the fixed general SINDy library.
%
% The Stage-0 fixed library intentionally reuses the original independent
% general-SINDy generator from the attached v69 package.  Therefore the default
% library contains
%   1) the constant term;
%   2) every polynomial monomial of the raw inputs up to opts.polyOrder;
%   3) the configured unary-operator terms on every input dimension and, when
%      enabled by the original SINDy options, on pure-power monomials; and
%   4) the original optional variable-times-unary cross terms.
%
% No evolutionary-backend-specific simplified dictionary is constructed here.  Any fields in
% options.baseDictionary only override the original general-SINDy defaults.

    t = tic;

    % Start from the exact independent general-SINDy baseline defaults.  This
    % keeps Stage 0 and the SINDy baseline aligned without modifying the
    % baseline implementation itself.
    sopts = make_default_sindy_options_for_demo();
    if nargin >= 4 && isstruct(options) && isfield(options, 'baseDictionary') && ...
            isstruct(options.baseDictionary)
        cfg = options.baseDictionary;
        names = fieldnames(cfg);
        for i = 1:numel(names)
            sopts.(names{i}) = cfg.(names{i});
        end
    end
    sopts.dictionaryMode = 'general';
    sopts.usePhdnDictionarySupport = false;

    baseArch = make_sindy_general_arch(task, sopts, archBase);

    [PhiTr, termNames, invalidTr] = eval_arch_local(data.Xtr, baseArch);
    [PhiVal, ~, invalidVal] = eval_arch_local(data.Xval, baseArch);
    [PhiTe, ~, invalidTe] = eval_arch_local(data.Xte, baseArch);
    if isfield(data, 'Xood') && ~isempty(data.Xood)
        [PhiOod, ~, invalidOod] = eval_arch_local(data.Xood, baseArch);
    else
        PhiOod = [];
        invalidOod = false(size(invalidTr));
    end

    % Structure selection must not inspect test or OOD behavior.  Only the
    % training and validation sets determine whether a basis is admissible.
    keep = ~(invalidTr | invalidVal);
    keep = keep & all(isfinite(PhiTr), 1).' & all(isfinite(PhiVal), 1).';
    PhiTr = PhiTr(:, keep);
    PhiVal = PhiVal(:, keep);
    PhiTe = PhiTe(:, keep);
    if ~isempty(PhiOod); PhiOod = PhiOod(:, keep); end
    termNames = termNames(keep);

    % Remove semantic/numerical redundancy already present inside the broad
    % general-SINDy library before fitting the fast path. This prevents terms
    % such as x*inv(x) from surviving alongside the constant column.
    preliminary = struct('termNames',{termNames(:)},'PhiTr',PhiTr,'PhiVal',PhiVal, ...
        'PhiTe',PhiTe,'PhiOod',PhiOod,'nTerms',numel(termNames));
    pruneOptions = struct('spanRedundancyTolerance', ...
        getfield_default_local(options,'spanRedundancyTolerance',1e-10), ...
        'minimumColumnNorm',1e-12);
    [preliminary, redundancyInfo] = prune_stage0_dictionary_columns(preliminary,pruneOptions);
    termNames = preliminary.termNames; PhiTr = preliminary.PhiTr;
    PhiVal = preliminary.PhiVal; PhiTe = preliminary.PhiTe; PhiOod = preliminary.PhiOod;

    base = struct();
    base.arch = baseArch;
    base.options = sopts;
    base.termNames = termNames(:);
    base.PhiTr = PhiTr;
    base.PhiVal = PhiVal;
    base.PhiTe = PhiTe;
    base.PhiOod = PhiOod;
    base.nTerms = numel(termNames);
    base.redundancyPruning = redundancyInfo;
    base.invalidRows = struct('train', invalidTr, 'validation', invalidVal, ...
        'test', invalidTe, 'ood', invalidOod, 'keep', keep);
    elapsedTime = toc(t);
end

function [Phi, names, invalidRows] = eval_arch_local(X, arch)
    branch = build_branch_cache(X.', arch, 1, {});
    Phi = branch.Phi.';
    invalidRows = branch.PhiInvalidRows(:);
    info = branch_dictionary_terms(size(X, 2), arch, 'x', 1);
    if isstruct(info) && isfield(info, 'name')
        names = {info.name}.';
    elseif iscell(info)
        names = info(:);
    else
        names = arrayfun(@(k) sprintf('phi%d', k), 1:size(Phi,2), 'UniformOutput', false).';
    end
end

function value = getfield_default_local(s,name,defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name)); value=s.(name); else; value=defaultValue; end
end
