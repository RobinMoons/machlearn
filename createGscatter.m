function plots = createGscatter(windowTitle, m_tr, c_tr, m_te, c_te, model_1, m_tr2, c_tr2, m_te2, c_te2, model_2)
    d = 0.01;
    [x1Grid,x2Grid] = meshgrid(min(m_tr(:,1)):d:max(m_tr(:,1)), min(m_tr(:,2)):d:max(m_tr(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    labels = predict(model_1,xGrid);
    % Training data points
    plots = figure('Name', windowTitle, 'NumberTitle', 'off')
    subplot(2,2,1);
    h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
    title('training data points 1')
    hold on
    h(3:4) = gscatter(m_tr(:,1),m_tr(:,2),c_tr);
    legend(h,{'1 tr','2 tr','1 te','2 te'},'Location','Northwest');
    % Testing data points
    [x1Grid,x2Grid] = meshgrid(min(m_te(:,1)):d:max(m_te(:,1)), min(m_te(:,2)):d:max(m_te(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    labels = predict(model_1,xGrid);
    subplot(2,2,2);
    h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
    title('test data points 1')
    hold on
    h(3:4) = gscatter(m_te(:,1),m_te(:,2),c_te);
    legend(h,{'1 tr','2 tr','1 te','2 te'},'Location','Northwest');
    d = 0.01;
    [x1Grid,x2Grid] = meshgrid(min(m_tr2(:,1)):d:max(m_tr2(:,1)), min(m_tr2(:,2)):d:max(m_tr2(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    labels = predict(model_2,xGrid);
    % Training data points
    subplot(2,2,3);
    h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
    title('training data points 2')
    hold on
    h(3:4) = gscatter(m_tr2(:,1),m_tr2(:,2),c_tr2);
    legend(h,{'1 tr','2 tr','1 te','2 te'},'Location','Northwest');
    % Testing data points
    [x1Grid,x2Grid] = meshgrid(min(m_te2(:,1)):d:max(m_te2(:,1)), min(m_te2(:,2)):d:max(m_te2(:,2)));
    xGrid = [x1Grid(:),x2Grid(:)];
    labels = predict(model_2,xGrid);
    subplot(2,2,4);
    h(1:2) = gscatter(xGrid(:,1),xGrid(:,2),labels,[0.1 0.5 0.5; 0.5 0.1 0.5 ]);
    title('test data points 2')
    hold on
    h(3:4) = gscatter(m_te2(:,1),m_te2(:,2),c_te2);
    legend(h,{'1 tr','2 tr','1 te','2 te'},'Location','Northwest');
end