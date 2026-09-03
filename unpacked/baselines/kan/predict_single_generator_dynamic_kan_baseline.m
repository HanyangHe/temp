function Y = predict_single_generator_dynamic_kan_baseline(result,X)
%PREDICT_SINGLE_GENERATOR_DYNAMIC_KAN_BASELINE Portable selected pyKAN inference.
    if isvector(X); X=reshape(X,1,[]); end
    if ~isfield(result,'portableModel') || isempty(fieldnames(result.portableModel))
        error('Selected KAN portable model is unavailable.');
    end
    model=result.portableModel;
    normInfo=model.normalization;
    xMean=unpack_array_local(normInfo.x_mean);
    xStd=unpack_array_local(normInfo.x_std);
    yMean=unpack_array_local(normInfo.y_mean);
    yStd=unpack_array_local(normInfo.y_std);
    x=(double(X)-reshape(xMean,1,[]))./reshape(xStd,1,[]);
    inputId=double(model.input_id_zero_based(:).')+1;
    x=x(:,inputId);
    layers=normalize_struct_list_local(model.layers);
    for ell=1:numel(layers)
        layer=layers{ell};
        grid=unpack_array_local(layer.grid);
        coef=unpack_array_local(layer.coef);
        scaleBase=unpack_array_local(layer.scale_base);
        scaleSpline=unpack_array_local(layer.scale_spline);
        mask=unpack_array_local(layer.mask);
        subScale=reshape(unpack_array_local(layer.subnode_scale),1,[]);
        subBias=reshape(unpack_array_local(layer.subnode_bias),1,[]);
        nodeScale=reshape(unpack_array_local(layer.node_scale),1,[]);
        nodeBias=reshape(unpack_array_local(layer.node_bias),1,[]);
        k=double(layer.spline_order);
        [nSample,nIn]=size(x); nOut=size(scaleBase,2);
        ySub=zeros(nSample,nOut);
        switch lower(char(layer.base_function))
            case 'silu'
                base=x.*stable_sigmoid_local(x);
            case 'identity'
                base=x;
            case 'zero'
                base=zeros(size(x));
            otherwise
                error('Unsupported portable KAN base function: %s',char(layer.base_function));
        end
        for i=1:nIn
            basis=bspline_basis_1d_local(x(:,i),grid(i,:),k);
            c=reshape(coef(i,:,:),nOut,size(coef,3));
            splineValue=basis*c.';
            ySub=ySub + ...
                (base(:,i).*reshape(scaleBase(i,:),1,[]) + ...
                 splineValue.*reshape(scaleSpline(i,:),1,[])).* ...
                 reshape(mask(i,:),1,[]);
        end
        ySub=ySub.*subScale+subBias;
        nSum=double(layer.sum_node_count);
        nMult=double(layer.multiplication_node_count);
        if nMult>0
            arities=double(layer.multiplication_arities(:).');
            if numel(arities)~=nMult
                error('Portable KAN multiplication arity metadata is inconsistent.');
            end
            xNext=zeros(nSample,nSum+nMult);
            if nSum>0; xNext(:,1:nSum)=ySub(:,1:nSum); end
            cursor=nSum+1;
            for j=1:nMult
                ids=cursor:(cursor+arities(j)-1);
                xNext(:,nSum+j)=prod(ySub(:,ids),2);
                cursor=cursor+arities(j);
            end
        else
            xNext=ySub(:,1:nSum);
        end
        x=xNext.*nodeScale+nodeBias;
    end
    Y=x.*reshape(yStd,1,[])+reshape(yMean,1,[]);
end

function A=unpack_array_local(payload)
    shape=double(payload.shape(:).');
    data=double(payload.data(:));
    if isempty(shape)
        A=data;
    elseif numel(shape)==1
        A=reshape(data,shape(1),1);
    else
        A=reshape(data,shape);
    end
end

function items=normalize_struct_list_local(raw)
    if iscell(raw); items=raw(:).'; return; end
    items=cell(1,numel(raw));
    for i=1:numel(raw); items{i}=raw(i); end
end

function y=stable_sigmoid_local(x)
    y=zeros(size(x)); pos=x>=0;
    y(pos)=1./(1+exp(-x(pos)));
    ex=exp(x(~pos)); y(~pos)=ex./(1+ex);
end

function B=bspline_basis_1d_local(x,knots,k)
    x=double(x(:)); knots=double(knots(:).');
    nBasis0=numel(knots)-1;
    B=false(numel(x),nBasis0);
    for j=1:nBasis0
        B(:,j)=x>=knots(j) & x<knots(j+1);
    end
    B=double(B);
    for order=1:k
        next=zeros(numel(x),nBasis0-order);
        for j=1:(nBasis0-order)
            leftDen=knots(j+order)-knots(j);
            rightDen=knots(j+order+1)-knots(j+1);
            left=zeros(size(x)); right=zeros(size(x));
            if leftDen~=0
                left=((x-knots(j))/leftDen).*B(:,j);
            end
            if rightDen~=0
                right=((knots(j+order+1)-x)/rightDen).*B(:,j+1);
            end
            next(:,j)=left+right;
        end
        next(~isfinite(next))=0;
        B=next;
    end
end
