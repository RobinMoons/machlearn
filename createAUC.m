function [plots, AUC1, AUC2, AUC3, AUC4] = createAUC( s_tr, c_tr, s_te, c_te, s_tr2, c_tr2, s_te2, c_te2)
    [fpr,tpr,T,AUC,OPTROCPT] = perfcurve(c_tr,s_tr(:,1),1);
    AUC1 = AUC;
    plots = figure('Name', 'ROC curves', 'NumberTitle', 'off')
    subplot(2,2,1);
    plot(fpr,tpr)
    hold on
    plot(OPTROCPT(1),OPTROCPT(2),'ro')
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('training data points 1')
    hold off
    
    [fpr,tpr,T,AUC,OPTROCPT] = perfcurve(c_te,s_te(:,1),1);        
    AUC2 = AUC;
    subplot(2,2,2);
    plot(fpr,tpr)
    hold on
    plot(OPTROCPT(1),OPTROCPT(2),'ro')
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('test data points 1')
    hold off
    
    [fpr,tpr,T,AUC,OPTROCPT] = perfcurve(c_tr2,s_tr2(:,1),1);
    AUC3 = AUC;
    subplot(2,2,3);    
    plot(fpr,tpr)
    hold on
    plot(OPTROCPT(1),OPTROCPT(2),'ro')
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('training data points 2')
    hold off
    
    [fpr,tpr,T,AUC,OPTROCPT] = perfcurve(c_te2,s_te2(:,1),1);
    subplot(2,2,4);
    AUC4 = AUC;
    plot(fpr,tpr)
    hold on
    plot(OPTROCPT(1),OPTROCPT(2),'ro')
    xlabel('False positive rate')
    ylabel('True positive rate')
    title('test data points 2')
    hold off
end