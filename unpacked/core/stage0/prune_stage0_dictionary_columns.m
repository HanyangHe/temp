function [dictionary, info] = prune_stage0_dictionary_columns(dictionary, options)
%PRUNE_STAGE0_DICTIONARY_COLUMNS Remove exact/numerical semantic redundancy.
%
% Columns are processed in their original deterministic order. A column is
% removed when it is nonfinite, nearly zero, a signed duplicate, or lies in the
% numerical span of previously accepted columns. This catches identities such
% as x*inv(x)=1 even when they originate in the fixed general-SINDy library.

    tol = get_option_local(options, 'spanRedundancyTolerance', 1e-10);
    minNorm = get_option_local(options, 'minimumColumnNorm', 1e-12);
    A = dictionary.PhiTr;
    keep = false(1,size(A,2));
    Q = zeros(size(A,1),0);
    removedReason = strings(size(A,2),1);

    for j = 1:size(A,2)
        v = A(:,j);
        nv = norm(v);
        if ~all(isfinite(v)) || nv <= minNorm
            removedReason(j) = "nonfinite_or_zero";
            continue;
        end
        if isempty(Q)
            residual = v;
        else
            residual = v - Q*(Q'*v);
        end
        relResidual = norm(residual) / max(nv, eps);
        if relResidual <= tol
            removedReason(j) = "in_previous_span";
            continue;
        end
        keep(j) = true;
        q = residual / norm(residual);
        % One reorthogonalization pass improves stability for wide libraries.
        if ~isempty(Q)
            q = q - Q*(Q'*q);
            nq = norm(q);
            if nq <= tol
                keep(j) = false;
                removedReason(j) = "in_previous_span_reorth";
                continue;
            end
            q = q/nq;
        end
        Q(:,end+1) = q; %#ok<AGROW>
    end

    fields = {'termNames','PhiTr','PhiVal','PhiTe','PhiOod'};
    for i = 1:numel(fields)
        f = fields{i};
        if ~isfield(dictionary,f) || isempty(dictionary.(f)); continue; end
        if strcmp(f,'termNames')
            dictionary.(f) = dictionary.(f)(keep);
        else
            dictionary.(f) = dictionary.(f)(:,keep);
        end
    end
    dictionary.nTerms = nnz(keep);
    info = struct('keepMask',keep(:),'removedCount',nnz(~keep), ...
        'removedReason',removedReason,'tolerance',tol);
end

function value = get_option_local(s,name,defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name)); value=s.(name); else; value=defaultValue; end
end
