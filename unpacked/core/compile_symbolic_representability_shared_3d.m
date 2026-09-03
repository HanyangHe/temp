function [arch, CoefExact, compileInfo] = compile_symbolic_representability_shared_3d(task, symbolicExpressions)
%COMPILE_SYMBOLIC_REPRESENTABILITY_SHARED_3D Exact shared-subexpression DAG.
%
% Layer dimension vector: [3, 4, 3, 4, 4, 3, 1].

    if nargin < 1 || isempty(task)
        task = task_symbolic_representability_shared_3d();
    end
    if nargin < 2 || isempty(symbolicExpressions)
        symbolicExpressions = task.symbolicExpressions;
    end

    L = 6;
    branchActive = false(L,L);
    for ell = 1:L
        branchActive(1,ell) = true;
    end

    termsByBlock = cell(L,L);
    rowTerms = cell(L,L);
    rowSeedCoef = cell(L,L);

    % h2 = [x1^2+x2^2+0.5*x3^2, x1*x2, 1+x3^2, 1+x3].
    termsByBlock{1,1} = {'1','v1^2','v2^2','v3^2','v1*v2','v3'};
    rowTerms{1,1} = { ...
        {'v1^2','v2^2','v3^2'}, ...
        {'v1*v2'}, ...
        {'1','v3^2'}, ...
        {'1','v3'} ...
    };
    rowSeedCoef{1,1} = {[1,1,0.5],1,[1,1],[1,1]};

    % h3 = [sqrt(h2_1), h2_2*inv(h2_3), inv(h2_4)].
    termsByBlock{1,2} = {'sqrt(v1)','v2*inv(v3)','inv(v4)'};
    rowTerms{1,2} = {{'sqrt(v1)'},{'v2*inv(v3)'},{'inv(v4)'}};
    rowSeedCoef{1,2} = {1,1,1};

    % h4 = [exp(r), cos(q), r+q, inv(1+x3)].
    termsByBlock{1,3} = {'exp(v1)','cos(v2)','v1','v2','v3'};
    rowTerms{1,3} = { ...
        {'exp(v1)'}, ...
        {'cos(v2)'}, ...
        {'v1','v2'}, ...
        {'v3'} ...
    };
    rowSeedCoef{1,3} = {1,1,[1,1],1};

    % h5 = [1+exp(r), cos(q), sin(r+q), inv(1+x3)].
    termsByBlock{1,4} = {'1','v1','v2','sin(v3)','v4'};
    rowTerms{1,4} = { ...
        {'1','v1'}, ...
        {'v2'}, ...
        {'sin(v3)'}, ...
        {'v4'} ...
    };
    rowSeedCoef{1,4} = {[1,1],1,1,1};

    % h6 = [log(1+exp(r)), cos(q), sin(r+q)/(1+x3)].
    termsByBlock{1,5} = {'log(v1)','v2','v3*v4'};
    rowTerms{1,5} = {{'log(v1)'},{'v2'},{'v3*v4'}};
    rowSeedCoef{1,5} = {1,1,1};

    % y = h6_1+h6_2+h6_3.
    termsByBlock{1,6} = {'v1','v2','v3'};
    rowTerms{1,6} = {{'v1','v2','v3'}};
    rowSeedCoef{1,6} = {[1,1,1]};

    spec = struct();
    spec.caseLabel = 'Case 3: three-input scalar DAG with reusable r and q channels';
    spec.layer = L;
    spec.hiddenDims = [4,3,4,4,3];
    spec.branchActiveMask = branchActive;
    spec.branchActiveMode = 'constructive_shared_subexpression_chain';
    spec.source = 'Manual exact compilation of the shared 3-D scalar expression';
    spec.termsByBlock = termsByBlock;
    spec.rowTerms = rowTerms;
    spec.rowSeedCoef = rowSeedCoef;
    spec.sharedSubexpressions = { ...
        'r=sqrt(x1^2+x2^2+0.5*x3^2)', ...
        'q=x1*x2/(1+x3^2)' ...
    };
    spec.nodeEquations = { ...
        'h2 = [x1^2+x2^2+0.5*x3^2, x1*x2, 1+x3^2, 1+x3]', ...
        'h3 = [sqrt(h2_1), h2_2*inv(h2_3), inv(h2_4)] = [r,q,inv(1+x3)]', ...
        'h4 = [exp(h3_1), cos(h3_2), h3_1+h3_2, h3_3]', ...
        'h5 = [1+h4_1, h4_2, sin(h4_3), h4_4]', ...
        'h6 = [log(h5_1), h5_2, h5_3*h5_4]', ...
        'y = h6_1+h6_2+h6_3' ...
    };
    spec.summary = ['The reusable channels r and q each feed two downstream ', ...
        'operations, producing a genuine six-layer DAG rather than a simple tree.'];

    [arch, CoefExact, compileInfo] = build_manual_constructive_phdn( ...
        task, symbolicExpressions, spec);
end
