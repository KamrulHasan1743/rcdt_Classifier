clear all;close all; clc;
tic
load 'feat.mat'
labl(labl>10)=labl(labl>10)-1;
k_fold=10;
indice= crossvalind('Kfold',labl,k_fold);
GT=labl;


pred_all=zeros(length(GT),1);
for i=1:k_fold
    tt_id=(indice==i);
    tr_id=~tt_id;
    
    Xtrain=feat(tr_id,:)';
    Y_train=GT(tr_id);
    
    Xtest=feat(tt_id,:)';
    Y_test=GT(tt_id);
    trainSamples=[4 8 12 16 20 24 28 32 36];
    for j=1:length(trainSamples)
        len_subspace = 0;
        for cls=[1:9 10:max(labl)]
            ind = find(Y_train==cls);           % find train samples corresponding to class 'cls'
            ind_sub = randsample(ind,trainSamples(j)); % control the number of train samples to fit the model using
            %         ind_sub = ind;                            % 'trainSamples' variable; all the samples can also be used
            % by setting 'ind_sub = ind'
            classSamples = Xtrain(:,ind_sub);
            
            % calculate basis vectors using SVD
            [uu,su,vu]=svds(classSamples,size(classSamples,2));
            s=diag(su);
            eps= 1e-4;
            indx=find(s>eps);
            V=uu(:,indx);
            basis(cls).V = V;
            % take basis components with atleast 99% variance
            S = cumsum(s);
            S = S/max(S);
            basis_ind = find(S>=0.99);
            if len_subspace < basis_ind(1)
                len_subspace = basis_ind(1);  % len_subspace is max over all classes
            end
        end
        
        %% PREDICT: classify the test samples
        D=[];
        Dproj=[];
        
        for cls=1:max(labl)
            B = basis(cls).V;
            B = B(:,1:len_subspace);
            Xproj = B*(B'*Xtest);               % projection of the test sample on the subspace
            Dproj = Xtest - Xproj;
            D(cls,:) = sqrt(sum(Dproj.^2,1));
        end
        [~,Ypred] = min(D);                     % predict the class label of the test sample
        CM = confusionmat(Y_test,Ypred)
        Accuracy(i,j) = numel(find(Y_test==Ypred))/length(Y_test)
        
    end
end

MeanAccuracy_with_no_of_trainsamples=[trainSamples;mean(Accuracy)]
figure, bar(trainSamples,mean(Accuracy));
xlabel('No of samples per class in Training')
ylabel('Accuracy')
