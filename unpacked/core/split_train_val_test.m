function [Xtr, Ytr, Xval, Yval, Xte, Yte] = split_train_val_test(X, Y, ratioTrain, ratioVal)
%SPLIT_TRAIN_VAL_TEST Random train/validation/test split.

	N = size(X, 1);
	idx = randperm(N);

	nTrain = round(ratioTrain * N);
	nVal = round(ratioVal * N);

	idTrain = idx(1:nTrain);
	idVal = idx(nTrain + 1 : nTrain + nVal);
	idTest = idx(nTrain + nVal + 1 : end);

	Xtr = X(idTrain, :);
	Ytr = Y(idTrain, :);
	Xval = X(idVal, :);
	Yval = Y(idVal, :);
	Xte = X(idTest, :);
	Yte = Y(idTest, :);
end
