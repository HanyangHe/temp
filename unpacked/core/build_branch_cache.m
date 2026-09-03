function branch = build_branch_cache(H, arch, layerIndex, hContext, branchIndex)
%BUILD_BRANCH_CACHE Build explicit case-specific branch dictionary cache.
%
% previous-version clean dictionary rule:
%   Phi is evaluated only from the explicit case dictionary supplied by the
%   task/arch.  No unified polynomial/order/operator/cross enumeration is
%   performed in previous-version.

	if nargin < 3
		layerIndex = [];
	end
	if nargin < 4
		hContext = {}; %#ok<NASGU>
	end
	if nargin < 5
		branchIndex = [];
	end
	if ~isfield(arch, 'safety') || isempty(arch.safety)
		arch.safety = struct();
	end
	if ~isfield(arch.safety, 'eps') || isempty(arch.safety.eps)
		arch.safety.eps = 1e-8;
	end

	termNames = explicit_case_dictionary_terms(size(H, 1), arch, layerIndex, branchIndex);
	rawPySRSemanticsRows = structural_pysr_semantics_mask_local(termNames, arch, layerIndex, branchIndex);
	[Phi, Jphi, PhiInvalidRows, termNames] = evaluate_explicit_case_terms( ...
		H, termNames, arch.safety, arch, rawPySRSemanticsRows);
	PhiInvalidRows = PhiInvalidRows | any(~isfinite(Phi), 2);
	Phi(~isfinite(Phi)) = 0;
	Jphi(~isfinite(Jphi)) = 0;

	branch = struct();
	branch.H = H;
	branch.HclipMask = false(size(H));
	branch.termNames = termNames(:);
	branch.Phi = Phi;
	branch.Jphi = Jphi;
	branch.PhiInvalidRows = PhiInvalidRows;
	branch.PhiValidRows = ~PhiInvalidRows;
	branch.rawPySRSemanticsRows = rawPySRSemanticsRows(:);
	branch.layerIndex = layerIndex;

	% Compatibility fields for legacy reporting/backprop code.  They are empty
	% because previous-version explicit dictionaries do not have generated U/G/cross blocks.
	branch.cfg = struct('mode', 'case_specific_explicit', 'opNames', {{}});
	branch.U = zeros(0, size(H, 2));
	branch.Ju = zeros(0, size(H, 1), size(H, 2));
	branch.Eu = zeros(0, size(H, 1));
	branch.Q = zeros(0, size(H, 2));
	branch.Jq = zeros(0, size(H, 1), size(H, 2));
	branch.Eq = zeros(0, size(H, 1));
	branch.G = zeros(0, size(H, 2));
	branch.dOp = {};
	branch.GinvalidSamples = false(0, size(H, 2));
	branch.Gdegree = zeros(0, 1);
	branch.GopIndex = zeros(0, 1);
	branch.GqIndex = zeros(0, 1);
	branch.Uleft = zeros(0, size(H, 2));
	branch.Jleft = zeros(0, size(H, 1), size(H, 2));
	branch.Eleft = zeros(0, size(H, 1));
	% Expose the explicit linear dictionary rows v1,...,vd as legacy baseU rows.
	% These indices remain useful for diagnostics and symbolic display.
	baseURows = nan(1, size(H, 1));
	for kk = 1:size(H, 1)
		idxK = find(strcmp(termNames, sprintf('v%d', kk)), 1);
		if ~isempty(idxK)
			baseURows(kk) = idxK;
		end
	end
	baseURows = baseURows(isfinite(baseURows));
	branch.idx = struct('constant', find(strcmp(termNames, '1'), 1), 'baseU', baseURows, 'baseG', []);
	branch.cross = struct('leftIndex', [], 'rightIndex', [], 'rows', [], 'invalidRows', []);
	branch.invalidReport = make_report_local(PhiInvalidRows);
end


function mask = structural_pysr_semantics_mask_local(termNames, arch, layerIndex, branchIndex)
%STRUCTURAL_PYSR_SEMANTICS_MASK_LOCAL Mark Stage-0 structural terms.
%
% The SR structural channel must reproduce the selected official-PySR
% expression without PhDN denominator/domain protection.  The separately
% appended constant+polynomial augmentation channel retains the ordinary PhDN
% evaluator semantics.  A term is marked only when the compiled dictionary
% explicitly declares official-PySR raw structural semantics.
	mask = false(numel(termNames), 1);
	if isempty(termNames) || isempty(layerIndex) || isempty(branchIndex) || ...
			~isfield(arch, 'caseDictionary') || ~isstruct(arch.caseDictionary)
		return;
	end
	D = arch.caseDictionary;
	if ~isfield(D, 'structuralOperatorSemantics') || ...
			~strcmpi(strtrim(char(D.structuralOperatorSemantics)), 'official_pysr_raw')
		return;
	end
	if ~isfield(D, 'structuralTermsByBlock') || ~iscell(D.structuralTermsByBlock) || ...
			size(D.structuralTermsByBlock, 1) < branchIndex || ...
			size(D.structuralTermsByBlock, 2) < layerIndex
		return;
	end
	structural = D.structuralTermsByBlock{branchIndex, layerIndex};
	if isempty(structural)
		return;
	end
	if ischar(structural) || isstring(structural)
		structural = cellstr(structural);
	end
	structural = cellfun(@normalize_term_text_local, structural(:), 'UniformOutput', false);
	for k = 1:numel(termNames)
		mask(k) = any(strcmp(structural, normalize_term_text_local(termNames{k})));
	end
end

function out = normalize_term_text_local(in)
	out = strrep(strtrim(char(in)), ' ', '');
end

function report = make_report_local(PhiInvalidRows)
	report = struct();
	report.nPhiRows = numel(PhiInvalidRows);
	report.nValidRows = nnz(~PhiInvalidRows);
	report.nInvalidRows = nnz(PhiInvalidRows);
	report.invalidRows = find(PhiInvalidRows(:)).';
	report.nInvalidConstantRows = 0;
	report.nInvalidBasePolyRows = 0;
	report.nInvalidBaseOpRows = 0;
	report.nInvalidCrossRows = 0;
	report.invalidCrossRows = [];
	report.nInvalidMutualRows = 0;
	report.invalidMutualRows = [];
end
