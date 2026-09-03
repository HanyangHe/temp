function [arch, CoefExact, compileInfo] = compile_symbolic_representability_dense_skip_2d(task, symbolicExpressions)
%COMPILE_SYMBOLIC_REPRESENTABILITY_DENSE_SKIP_2D Moderate 2-D skip DAG.
%
% Layer dimension vector: [2, 2, 2, 2, 1].
% The case is intentionally easier to train than the previous dense v73m
% stress test, while still activating representative non-chain branches,
% including Coef_M{2,2}, Coef_M{2,3}, Coef_M{3,4}, and Coef_M{4,4}.

    if nargin < 1 || isempty(task)
        task = task_symbolic_representability_dense_skip_2d();
    end
    if nargin < 2 || isempty(symbolicExpressions)
        symbolicExpressions = task.symbolicExpressions;
    end

    L = 4;
    branchActive = false(L,L);
    branchActive(1,1) = true;
    branchActive(1,2) = true;
    branchActive(2,2) = true;
    branchActive(1,3) = true;
    branchActive(2,3) = true;
    branchActive(1,4) = true;
    branchActive(3,4) = true;
    branchActive(4,4) = true;

    termsByBlock = cell(L,L);
    rowTerms = cell(L,L);
    rowSeedCoef = cell(L,L);

    % h2 = [a,b], a=1+x1^2+x2^2, b=0.5*x1*x2.
    termsByBlock{1,1} = {'1','v1^2','v2^2','v1*v2'};
    rowTerms{1,1} = { ...
        {'1','v1^2','v2^2'}, ...
        {'v1*v2'} ...
    };
    rowSeedCoef{1,1} = {[1,1,1],0.5};

    % h3 = [u,v], immediate nonlinear transforms from h2.
    termsByBlock{1,2} = {'sqrt(v1)','exp(v2)'};
    rowTerms{1,2} = {{'sqrt(v1)'},{'exp(v2)'}};
    rowSeedCoef{1,2} = {1,1};

    % Direct input contribution x -> h3, i.e. Coef_M{2,2}.
    termsByBlock{2,2} = {'sin(v1)','cos(v2)'};
    rowTerms{2,2} = {{'sin(v1)'},{'cos(v2)'}};
    rowSeedCoef{2,2} = {0.5,0.5};

    % h4 = [p,q], immediate nonlinear transforms from h3.
    termsByBlock{1,3} = {'log(v1)','sin(v2)'};
    rowTerms{1,3} = {{'log(v1)'},{'sin(v2)'}};
    rowSeedCoef{1,3} = {1,1};

    % h2 -> h4, i.e. Coef_M{2,3}.
    termsByBlock{2,3} = {'v1','v2'};
    rowTerms{2,3} = {{'v2'},{'v1'}};
    rowSeedCoef{2,3} = {0.5,0.5};

    % Output from h4.
    termsByBlock{1,4} = {'v1','v2'};
    rowTerms{1,4} = {{'v1','v2'}};
    rowSeedCoef{1,4} = {[1,0.5]};

    % h2 -> output, i.e. Coef_M{3,4}; use a distinct nonlinear feature.
    termsByBlock{3,4} = {'cos(v2)'};
    rowTerms{3,4} = {{'cos(v2)'}};
    rowSeedCoef{3,4} = {0.5};

    % Raw input -> output, i.e. Coef_M{4,4}.
    termsByBlock{4,4} = {'sin(v2)'};
    rowTerms{4,4} = {{'sin(v2)'}};
    rowSeedCoef{4,4} = {0.5};

    spec = struct();
    spec.caseLabel = 'Case 2: two-input moderate multi-source DAG';
    spec.layer = L;
    spec.hiddenDims = [2,2,2];
    spec.branchActiveMask = branchActive;
    spec.branchActiveMode = 'constructive_moderate_skip_selected_sources';
    spec.source = 'Manual exact compilation of the moderate-skip 2-D scalar expression';
    spec.termsByBlock = termsByBlock;
    spec.rowTerms = rowTerms;
    spec.rowSeedCoef = rowSeedCoef;
    spec.sharedSubexpressions = {'a','b','u','v','p','q'};
    spec.nodeEquations = { ...
        'h2 = [a,b] = [1+x1^2+x2^2, 0.5*x1*x2]', ...
        'h3_1 = u = sqrt(a)+0.5*sin(x1)', ...
        'h3_2 = v = exp(b)+0.5*cos(x2)', ...
        'h4_1 = p = log(u)+0.5*b', ...
        'h4_2 = q = sin(v)+0.5*a', ...
        'y = p+0.5*q+0.5*cos(b)+0.5*sin(x2)' ...
    };
    spec.summary = ['A low-width, four-stage PhDN with selected middle/skip ', ...
        'branches. Each non-chain branch contributes a distinct feature to ', ...
        'reduce parameter compensation during BP.'];

    [arch, CoefExact, compileInfo] = build_manual_constructive_phdn( ...
        task, symbolicExpressions, spec);
end
