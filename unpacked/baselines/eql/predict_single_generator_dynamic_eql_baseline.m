function Y = predict_single_generator_dynamic_eql_baseline(result,X)
%PREDICT_SINGLE_GENERATOR_DYNAMIC_EQL_BASELINE Portable selected EQL-Div inference.
    if isvector(X); X=reshape(X,1,[]); end
    if ~isfield(result,'portableModel') || isempty(fieldnames(result.portableModel))
        error('Selected EQL portable model is unavailable.');
    end
    model=result.portableModel; normInfo=model.normalization;
    xCenter=unpack_array_local(normInfo.x_center);
    xScale=unpack_array_local(normInfo.x_scale);
    yCenter=unpack_array_local(normInfo.y_center);
    yScale=unpack_array_local(normInfo.y_scale);
    x=(double(X)-reshape(xCenter,1,[]))./reshape(xScale,1,[]);
    hidden=normalize_struct_list_local(model.hidden_layers);
    for ell=1:numel(hidden)
        layer=hidden{ell};
        W=unpack_array_local(layer.weights); b=reshape(unpack_array_local(layer.bias),1,[]);
        zall=x*W+b;
        unaryTypes=double(layer.unary_types(:).');
        binaryTypes=double(layer.binary_types(:).');
        nUnary=numel(unaryTypes); nBinary=numel(binaryTypes);
        zu=zall(:,1:nUnary);
        z1=zall(:,nUnary+(1:nBinary));
        z2=zall(:,nUnary+nBinary+(1:nBinary));
        unary=zeros(size(zu));
        for j=1:nUnary
            switch unaryTypes(j)
                case 0; unary(:,j)=zu(:,j);
                case 1; unary(:,j)=sin(zu(:,j));
                case 2; unary(:,j)=cos(zu(:,j));
                otherwise; error('Unsupported EQL unary type %g.',unaryTypes(j));
            end
        end
        binary=zeros(size(z1));
        for j=1:nBinary
            switch binaryTypes(j)
                case 0; binary(:,j)=z1(:,j).*z2(:,j);
                otherwise; error('Unsupported EQL binary type %g.',binaryTypes(j));
            end
        end
        x=[unary,binary];
    end
    out=model.output_layer;
    W=unpack_array_local(out.weights); b=reshape(unpack_array_local(out.bias),1,[]);
    z=x*W+b; nOut=size(z,2)/2;
    numerator=z(:,1:nOut); denominator=z(:,nOut+1:end);
    threshold=double(out.division_threshold);
    yNorm=zeros(size(numerator)); valid=denominator>=threshold;
    yNorm(valid)=numerator(valid)./denominator(valid);
    Y=yNorm.*reshape(yScale,1,[])+reshape(yCenter,1,[]);
end

function A=unpack_array_local(payload)
    shape=double(payload.shape(:).'); data=double(payload.data(:));
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
