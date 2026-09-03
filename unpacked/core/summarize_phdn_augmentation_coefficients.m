function info = summarize_phdn_augmentation_coefficients(Coef,mask,arch,threshold)
%SUMMARIZE_PHDN_AUGMENTATION_COEFFICIENTS Report newly added augmentation slots.
%
% Reporting only. A term already present in the exact SR structural dictionary
% is treated as an overlap, not as a newly added augmentation channel. This is
% important for constant/linear terms that can occur in both dictionaries.

    if nargin < 4 || isempty(threshold); threshold = 1e-10; end
    threshold = max(0,double(threshold));
    info = default_info_local(threshold);
    if ~isfield(arch,'caseDictionary') || ~isstruct(arch.caseDictionary) || ...
            ~isfield(arch.caseDictionary,'augmentationTermsByBlock') || ...
            ~iscell(arch.caseDictionary.augmentationTermsByBlock)
        return;
    end

    D = arch.caseDictionary;
    dims = get_arch_dims(arch);
    L = arch.layer;
    blockCounter = 0;
    sumSq = 0;
    typeSumSq = struct('neural',0,'linear',0,'constant',0,'other',0);

    for ell = 1:L
        for src = 1:ell
            if src>size(Coef,1) || ell>size(Coef,2) || isempty(Coef{src,ell})
                continue;
            end
            if src>size(D.augmentationTermsByBlock,1) || ...
                    ell>size(D.augmentationTermsByBlock,2) || ...
                    isempty(D.augmentationTermsByBlock{src,ell})
                continue;
            end
            stateIndex = ell-src+1;
            termNames = normalize_cellstr_local( ...
                explicit_case_dictionary_terms(dims(stateIndex),arch,ell,src));
            augTerms = normalize_cellstr_local(D.augmentationTermsByBlock{src,ell});
            structuralTerms = {};
            if isfield(D,'structuralTermsByBlock') && iscell(D.structuralTermsByBlock) && ...
                    src<=size(D.structuralTermsByBlock,1) && ...
                    ell<=size(D.structuralTermsByBlock,2) && ...
                    ~isempty(D.structuralTermsByBlock{src,ell})
                structuralTerms = normalize_cellstr_local(D.structuralTermsByBlock{src,ell});
            end
            if numel(termNames)~=size(Coef{src,ell},2)
                continue;
            end

            isAug = ismember(termNames,augTerms);
            isOverlap = isAug & ismember(termNames,structuralTerms);
            isNewAug = isAug & ~isOverlap;
            A = Coef{src,ell};
            M = true(size(A));
            if nargin>=2 && ~isempty(mask) && src<=size(mask,1) && ell<=size(mask,2) && ...
                    ~isempty(mask{src,ell}) && isequal(size(mask{src,ell}),size(A))
                M = logical(mask{src,ell});
            end

            info.overlapColumns = info.overlapColumns + nnz(isOverlap);
            for c = find(isNewAug(:).')
                term = termNames{c};
                type = classify_term_local(term);
                vals = A(:,c);
                active = M(:,c);
                activeVals = vals(active);
                nz = active & abs(vals)>threshold;
                info.declared = info.declared + size(A,1);
                info.trainable = info.trainable + nnz(active);
                info.nonzero = info.nonzero + nnz(nz);
                info.l1Norm = info.l1Norm + sum(abs(activeVals));
                sumSq = sumSq + sum(activeVals.^2);
                if ~isempty(activeVals)
                    info.maxAbs = max(info.maxAbs,max(abs(activeVals)));
                end
                prefix = type;
                info.([prefix 'Declared']) = info.([prefix 'Declared']) + size(A,1);
                info.([prefix 'Trainable']) = info.([prefix 'Trainable']) + nnz(active);
                info.([prefix 'Nonzero']) = info.([prefix 'Nonzero']) + nnz(nz);
                info.([prefix 'L1Norm']) = info.([prefix 'L1Norm']) + sum(abs(activeVals));
                typeSumSq.(prefix) = typeSumSq.(prefix) + sum(activeVals.^2);
                if ~isempty(activeVals)
                    info.([prefix 'MaxAbs']) = max(info.([prefix 'MaxAbs']),max(abs(activeVals)));
                end
            end

            blockCounter = blockCounter+1;
            info.blocks(blockCounter).src = src; %#ok<AGROW>
            info.blocks(blockCounter).ell = ell;
            info.blocks(blockCounter).newAugmentationColumns = nnz(isNewAug);
            info.blocks(blockCounter).overlapColumns = nnz(isOverlap);
            info.blocks(blockCounter).trainable = nnz(M(:,isNewAug));
            info.blocks(blockCounter).nonzero = nnz(M(:,isNewAug) & abs(A(:,isNewAug))>threshold);
        end
    end

    info.l2Norm = sqrt(sumSq);
    types = {'neural','linear','constant','other'};
    for k = 1:numel(types)
        info.([types{k} 'L2Norm']) = sqrt(typeSumSq.(types{k}));
    end
    info.available = true;
end

function info = default_info_local(threshold)
    info = struct('available',false,'declared',0,'trainable',0,'nonzero',0, ...
        'overlapColumns',0,'l1Norm',0,'l2Norm',0,'maxAbs',0,'threshold',threshold, ...
        'neuralDeclared',0,'neuralTrainable',0,'neuralNonzero',0, ...
        'neuralL1Norm',0,'neuralL2Norm',0,'neuralMaxAbs',0, ...
        'linearDeclared',0,'linearTrainable',0,'linearNonzero',0, ...
        'linearL1Norm',0,'linearL2Norm',0,'linearMaxAbs',0, ...
        'constantDeclared',0,'constantTrainable',0,'constantNonzero',0, ...
        'constantL1Norm',0,'constantL2Norm',0,'constantMaxAbs',0, ...
        'otherDeclared',0,'otherTrainable',0,'otherNonzero',0, ...
        'otherL1Norm',0,'otherL2Norm',0,'otherMaxAbs',0,'blocks',struct([]));
end

function type = classify_term_local(term)
    term = strrep(strtrim(char(term)),' ','');
    if strcmp(term,'1')
        type = 'constant';
    elseif ~isempty(regexp(term,'^v\d+$','once'))
        type = 'linear';
    elseif startsWith(term,'tanh(')
        type = 'neural';
    else
        type = 'other';
    end
end

function out = normalize_cellstr_local(in)
    if ischar(in) || isstring(in); in = cellstr(in); end
    out = cell(numel(in),1);
    for k = 1:numel(in)
        out{k} = strrep(strtrim(char(in{k})),' ','');
    end
    out = unique(out,'stable');
end
