function [arch, CoefExact, compileInfo] = compile_symbolic_representability_nested_2d(task, symbolicExpressions)
%COMPILE_SYMBOLIC_REPRESENTABILITY_NESTED_2D Exact deep 2-D scalar PhDN DAG.
%
% Layer dimension vector: [2, 3, 3, 3, 2, 2, 1].

    if nargin < 1 || isempty(task)
        task = task_symbolic_representability_nested_2d();
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

    % h2 = [x1^2+0.5*x2^2, 0.5*x1*x2, 1+x1^2+x2^2].
    termsByBlock{1,1} = {'1','v1^2','v2^2','v1*v2'};
    rowTerms{1,1} = { ...
        {'v1^2','v2^2'}, ...
        {'v1*v2'}, ...
        {'1','v1^2','v2^2'} ...
    };
    rowSeedCoef{1,1} = {[1,0.5], 0.5, [1,1,1]};

    % h3 = [exp(h2_1), exp(h2_2), inv(h2_3)].
    termsByBlock{1,2} = {'exp(v1)','exp(v2)','inv(v3)'};
    rowTerms{1,2} = {{'exp(v1)'},{'exp(v2)'},{'inv(v3)'}};
    rowSeedCoef{1,2} = {1,1,1};

    % h4 = [1+h3_1, sin(h3_2), h3_3].
    termsByBlock{1,3} = {'1','v1','sin(v2)','v3'};
    rowTerms{1,3} = {{'1','v1'},{'sin(v2)'},{'v3'}};
    rowSeedCoef{1,3} = {[1,1],1,1};

    % h5 = [log(h4_1), h4_2*h4_3].
    termsByBlock{1,4} = {'log(v1)','v2*v3'};
    rowTerms{1,4} = {{'log(v1)'},{'v2*v3'}};
    rowSeedCoef{1,4} = {1,1};

    % h6 = [0.25+h5_1, h5_2].
    termsByBlock{1,5} = {'1','v1','v2'};
    rowTerms{1,5} = {{'1','v1'},{'v2'}};
    rowSeedCoef{1,5} = {[0.25,1],1};

    % y = sqrt(h6_1)+h6_2.
    termsByBlock{1,6} = {'sqrt(v1)','v2'};
    rowTerms{1,6} = {{'sqrt(v1)','v2'}};
    rowSeedCoef{1,6} = {[1,1]};

    spec = struct();
    spec.caseLabel = 'Case 2: deep two-input scalar nested-composition DAG';
    spec.layer = L;
    spec.hiddenDims = [3,3,3,2,2];
    spec.branchActiveMask = branchActive;
    spec.branchActiveMode = 'constructive_deep_chain';
    spec.source = 'Manual exact compilation of the nested 2-D scalar expression';
    spec.termsByBlock = termsByBlock;
    spec.rowTerms = rowTerms;
    spec.rowSeedCoef = rowSeedCoef;
    spec.sharedSubexpressions = {'x1^2', 'x2^2'};
    spec.nodeEquations = { ...
        'h2 = [x1^2+0.5*x2^2, 0.5*x1*x2, 1+x1^2+x2^2]', ...
        'h3 = [exp(h2_1), exp(h2_2), inv(h2_3)]', ...
        'h4 = [1+h3_1, sin(h3_2), h3_3]', ...
        'h5 = [log(h4_1), h4_2*h4_3]', ...
        'h6 = [0.25+h5_1, h5_2]', ...
        'y = sqrt(h6_1)+h6_2' ...
    };
    spec.summary = ['A two-input scalar expression is represented by a six-layer ', ...
        'chain containing nested exp-log-sqrt composition and a separate ', ...
        'sin-exp-reciprocal product path.'];

    [arch, CoefExact, compileInfo] = build_manual_constructive_phdn( ...
        task, symbolicExpressions, spec);
end
