function [arch, CoefExact, compileInfo] = compile_symbolic_representability_dense_skip_3d(task, symbolicExpressions)
%COMPILE_SYMBOLIC_REPRESENTABILITY_DENSE_SKIP_3D Moderate 3-D two-output DAG.
%
% Layer dimension vector: [3, 3, 3, 2, 2].
% The case retains multi-source hidden layers and all four source depths at
% the output, including Coef_M{2,2}, Coef_M{2,3}, Coef_M{3,3},
% Coef_M{2,4}, Coef_M{3,4}, and Coef_M{4,4}, but removes one hidden stage
% and many compensating residual terms from the v73m stress test.

    if nargin < 1 || isempty(task)
        task = task_symbolic_representability_dense_skip_3d();
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
    branchActive(3,3) = true;
    branchActive(1,4) = true;
    branchActive(2,4) = true;
    branchActive(3,4) = true;
    branchActive(4,4) = true;

    termsByBlock = cell(L,L);
    rowTerms = cell(L,L);
    rowSeedCoef = cell(L,L);

    % h2 = [a,b,c].
    termsByBlock{1,1} = {'1','v1^2','v2^2','v3^2','v1*v2'};
    rowTerms{1,1} = { ...
        {'1','v1^2','v2^2'}, ...
        {'v1*v2'}, ...
        {'1','v3^2'} ...
    };
    rowSeedCoef{1,1} = {[1,1,1],0.5,[1,1]};

    % h3 = [u,v,w], immediate nonlinear transforms from h2.
    termsByBlock{1,2} = {'sqrt(v1)','exp(v2)','inv(v3)'};
    rowTerms{1,2} = {{'sqrt(v1)'},{'exp(v2)'},{'inv(v3)'}};
    rowSeedCoef{1,2} = {1,1,1};

    % Direct input contribution x -> h3, i.e. Coef_M{2,2}.
    termsByBlock{2,2} = {'sin(v1)','cos(v2)','v1*v3'};
    rowTerms{2,2} = {{'sin(v1)'},{'cos(v2)'},{'v1*v3'}};
    rowSeedCoef{2,2} = {0.5,0.5,0.5};

    % h4 = [p,q], immediate transforms from h3.
    termsByBlock{1,3} = {'log(v1)','sin(v2)','v3'};
    rowTerms{1,3} = { ...
        {'log(v1)'}, ...
        {'sin(v2)','v3'} ...
    };
    rowSeedCoef{1,3} = {1,[1,0.5]};

    % h2 -> h4, i.e. Coef_M{2,3}; only p uses b.
    termsByBlock{2,3} = {'v2'};
    rowTerms{2,3} = {{'v2'}, {}};
    rowSeedCoef{2,3} = {0.5, []};

    % x -> h4, i.e. Coef_M{3,3}.
    termsByBlock{3,3} = {'cos(v3)','v2^2'};
    rowTerms{3,3} = {{'cos(v3)'},{'v2^2'}};
    rowSeedCoef{3,3} = {0.5,0.5};

    % Two-output readout from h4.
    termsByBlock{1,4} = {'v1','v2'};
    rowTerms{1,4} = {{'v1','v2'},{'v1','v2'}};
    rowSeedCoef{1,4} = {[1,0.5],[0.5,1]};

    % h3 -> output, i.e. Coef_M{2,4}; only y2 uses u.
    termsByBlock{2,4} = {'v1'};
    rowTerms{2,4} = {{}, {'v1'}};
    rowSeedCoef{2,4} = {[], 0.5};

    % h2 -> output, i.e. Coef_M{3,4}; only y1 uses cos(b).
    termsByBlock{3,4} = {'cos(v2)'};
    rowTerms{3,4} = {{'cos(v2)'}, {}};
    rowSeedCoef{3,4} = {0.5, []};

    % Raw input -> output, i.e. Coef_M{4,4}.
    termsByBlock{4,4} = {'sin(v3)','cos(v1*v2)'};
    rowTerms{4,4} = {{'sin(v3)'},{'cos(v1*v2)'}};
    rowSeedCoef{4,4} = {0.5,0.5};

    spec = struct();
    spec.caseLabel = 'Case 3: three-input two-output moderate fan-in DAG';
    spec.layer = L;
    spec.hiddenDims = [3,3,2];
    spec.branchActiveMask = branchActive;
    spec.branchActiveMode = 'constructive_moderate_skip_multioutput';
    spec.source = 'Manual exact compilation of the moderate-skip 3-D two-output expression';
    spec.termsByBlock = termsByBlock;
    spec.rowTerms = rowTerms;
    spec.rowSeedCoef = rowSeedCoef;
    spec.sharedSubexpressions = {'a','b','c','u','v','w','p','q'};
    spec.nodeEquations = { ...
        'h2 = [a,b,c] = [1+x1^2+x2^2, 0.5*x1*x2, 1+x3^2]', ...
        'h3_1 = u = sqrt(a)+0.5*sin(x1)', ...
        'h3_2 = v = exp(b)+0.5*cos(x2)', ...
        'h3_3 = w = inv(c)+0.5*x1*x3', ...
        'h4_1 = p = log(u)+0.5*b+0.5*cos(x3)', ...
        'h4_2 = q = sin(v)+0.5*w+0.5*x2^2', ...
        'y1 = p+0.5*q+0.5*cos(b)+0.5*sin(x3)', ...
        'y2 = q+0.5*p+0.5*u+0.5*cos(x1*x2)' ...
    };
    spec.summary = ['A compact multi-output DAG with representative middle ', ...
        'branches and four source depths at the readout, but fewer layers and ', ...
        'less redundant residual fan-in than the previous stress test.'];

    [arch, CoefExact, compileInfo] = build_manual_constructive_phdn( ...
        task, symbolicExpressions, spec);
end
