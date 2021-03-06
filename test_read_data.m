close all;
clear all;
clc;

load('data.mat')

%%Plot first elements of array
% --> niet meer nodig, verder berekenen we resultantes
% --> eventueel te gebruiken in verslag
% drinking_x_firstMeasurement = data.drinking(1).x;
% drinking_y_firstMeasurement = data.drinking(1).y;
% drinking_z_firstMeasurement = data.drinking(1).z;
% 
% figure, plot(drinking_x_firstMeasurement)
% 
% brush_x_firstMeasurement = data.brush(1).x;
% brush_y_firstMeasurement = data.brush(1).y;
% brush_z_firstMeasurement = data.brush(1).z;
% 
% figure, plot(brush_x_firstMeasurement)
% 
% shoe_x_firstMeasurement = data.shoe(1).x;
% shoe_y_firstMeasurement = data.shoe(1).y;
% shoe_z_firstMeasurement = data.shoe(1).z;
% 
% figure, plot(shoe_x_firstMeasurement)
% 
% writing_x_firstMeasurement = data.writing(1).x;
% writing_y_firstMeasurement = data.writing(1).y;
% writing_z_firstMeasurement = data.writing(1).z;
% 
% figure, plot(writing_x_firstMeasurement)

%% Removing the mean from the sample
%edit Robin: dit lijkt te werken maar hebben we hier veel aan?
%eventueel moeten we dit voor alle waardes doen en daarna de resultantes
%berekenen?

% edit Bram: Heb het nog snel wat aangepast, mijn for lus ging nog niet tot
% de lengte van de array. Had ik fout over gekopie�rd.
% En ik denk dat ook. Overal mean aftrekken en dan x,y,z samentellen tot 1
% resulterende vector?
% Ik heb eens geprobeerd om dit te berekenen voor alle 15 meetresultaten
% van drinking. Je kijkt maar eens wat er van klopt of hoe ik dingen beter
% kan doen in matlab :)

%edit Robin: goed gewerkt, enkel de formule voor de resultante is normaal
%toch: wortel(x�+y�+z�) ??
%de subplots geven mooi weer wat we nu net gaan zoeken,laat ze dus maar 
%staan voor in het verslag, die man vind dat goed als ge u redenering laat
%zien en we kunnen dat wel gebruiken. 



% Calculate drinking
figure('NumberTitle','off','Name','Drinking data')
amountDrinking = numel(data.drinking);
for ii=1:amountDrinking
    % Get the x,y,z measurements
    drinking_x_Measurement = data.drinking(ii).x;
    drinking_y_Measurement = data.drinking(ii).y;
    drinking_z_Measurement = data.drinking(ii).z;
    % Calculate mean
    drinking_x_mean = mean(drinking_x_Measurement);
    drinking_y_mean = mean(drinking_y_Measurement);
    drinking_z_mean = mean(drinking_z_Measurement);
    % Resultant array
    N = numel(drinking_x_Measurement);
    drinking_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        drinking_x_Measurement(jj)=drinking_x_Measurement(jj)-drinking_x_mean;
        drinking_y_Measurement(jj)=drinking_y_Measurement(jj)-drinking_y_mean;
        drinking_z_Measurement(jj)=drinking_z_Measurement(jj)-drinking_z_mean;
        % Calculate resultant
        drinking_result(jj) = sqrt(drinking_x_Measurement(jj)^2 + drinking_y_Measurement(jj)^2 + drinking_z_Measurement(jj)^2);
    end
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(drinking_result)
    title(ii);
end


% Calculate brushing
figure('NumberTitle','off','Name','Brush data')
amountBrush = numel(data.brush);
for ii=1:amountBrush
    % Get the x,y,z measurements
    brush_x_Measurement = data.brush(ii).x;
    brush_y_Measurement = data.brush(ii).y;
    brush_z_Measurement = data.brush(ii).z;
    % Calculate mean
    brush_x_mean = mean(brush_x_Measurement);
    brush_y_mean = mean(brush_y_Measurement);
    brush_z_mean = mean(brush_z_Measurement);
    % Resultant array
    N = numel(brush_x_Measurement);
    brush_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        brush_x_Measurement(jj)=brush_x_Measurement(jj)-brush_x_mean;
        brush_y_Measurement(jj)=brush_y_Measurement(jj)-brush_y_mean;
        brush_z_Measurement(jj)=brush_z_Measurement(jj)-brush_z_mean;
        % Calculate resultant
        brush_result(jj) = sqrt(brush_x_Measurement(jj)^2 + brush_y_Measurement(jj)^2 + brush_z_Measurement(jj)^2);
    end
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(brush_result)
    title(ii);
end

% Calculate writing
figure('NumberTitle','off','Name','Writing data')
amountWriting = numel(data.writing);
for ii=1:amountWriting
    % Get the x,y,z measurements
    writing_x_Measurement = data.writing(ii).x;
    writing_y_Measurement = data.writing(ii).y;
    writing_z_Measurement = data.writing(ii).z;
    % Calculate mean
    writing_x_mean = mean(writing_x_Measurement);
    writing_y_mean = mean(writing_y_Measurement);
    writing_z_mean = mean(writing_z_Measurement);
    % Resultant array
    N = numel(writing_x_Measurement);
    writing_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        writing_x_Measurement(jj)=writing_x_Measurement(jj)-writing_x_mean;
        writing_y_Measurement(jj)=writing_y_Measurement(jj)-writing_y_mean;
        writing_z_Measurement(jj)=writing_z_Measurement(jj)-writing_z_mean;
        % Calculate resultant
        writing_result(jj) = sqrt(writing_x_Measurement(jj)^2 + writing_y_Measurement(jj)^2 + writing_z_Measurement(jj)^2);
    end
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(writing_result)
    title(ii);
end

% Calculate Shoe
amountShoe = numel(data.shoe);
figure('NumberTitle','off','Name','Shoe data')
for ii=1:amountShoe
    % Get the x,y,z measurements
    shoe_x_Measurement = data.shoe(ii).x;
    shoe_y_Measurement = data.shoe(ii).y;
    shoe_z_Measurement = data.shoe(ii).z;
    % Calculate mean
    shoe_x_mean = mean(shoe_x_Measurement);
    shoe_y_mean = mean(shoe_y_Measurement);
    shoe_z_mean = mean(shoe_z_Measurement);
    % Resultant array
    N = numel(shoe_x_Measurement);
    shoe_result = zeros(N,1);
    % Subtract mean from each value
    for jj=1:N
        shoe_x_Measurement(jj)=shoe_x_Measurement(jj)-shoe_x_mean;
        shoe_y_Measurement(jj)=shoe_y_Measurement(jj)-shoe_y_mean;
        shoe_z_Measurement(jj)=shoe_z_Measurement(jj)-shoe_z_mean;
        % Calculate resultant
        shoe_result(jj) = sqrt(shoe_x_Measurement(jj)^2 + shoe_y_Measurement(jj)^2 + shoe_z_Measurement(jj)^2);
    end
    % Plot resultant in 1 subplot
    subplot(4,4,ii)
    plot(shoe_result)
    title(ii);
end

%% Feature extraction

%Time domain:
% -mean amplitude
% array maken van de amplitudes -> mean berekenen
% -standard deviation amplitude
% standaardafwijking berekenen op array van amplitudes
% -skewness: https://nl.mathworks.com/help/stats/skewness.html 

%Frequency domain:
% -


%% Scatter plots van de features
%info voor scattered plot
%https://nl.mathworks.com/help/matlab/ref/scatter.html 


















