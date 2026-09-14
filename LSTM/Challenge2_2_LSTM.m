%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Challenge 2.2: Time-Forecasting of a pitching airfoil wake
%
% Challenge: The training set contains sets of 800 consecutive snapshots of
% flow over a pitching airfoil, at different average angles of attack and
% pitching frequency (16 different combinations).
% The test set contains cases both for the same conditions as the training
% data (though for future timesteps from the training data), and for
% parameter combinations not seen by the training data.
%
% For the testing data, a block of 70 consecutive snapshots is provided, 
% and the task is to forecast the next 130 snapshots.
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ML method: Long short-term memory (LSTM) networks
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script performs five steps:
%
% (1) Reads the training data and test input data
% (2) Identifies an LSTM model from the training data
% (3) Checks that the model produces reasonable results in estimating
%     portions of training dataset 
% (4) Estimates the unknown test output data using this model
% (5) Writes the result file, which serves as an example of the file
%     participants are expected to send to the challenge POC
%
% Datasets, challenge details and submission guidelines are
% maintained on the website:
% https://fluids-challenge.engin.umich.edu/
%
% 8/1/2026
% STMD <sdawson5@illinoistech.edu>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear variables
close all
clc


script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, 'utils'));
data_dir   = fullfile(script_dir, '..', 'data');

train_file       = fullfile(data_dir, 'Challenge2_2_train.h5');
test_input_file  = fullfile(data_dir, 'Challenge2_2_test_input.h5');
test_output_file = fullfile(data_dir, 'Challenge2_2_test_output.h5');

%% Parameters
x          = h5read(fullfile(data_dir, 'Challenge2_2_grid.h5'), '/x');
y          = h5read(fullfile(data_dir, 'Challenge2_2_grid.h5'), '/y');
dt         = h5read(fullfile(data_dir, 'Challenge2_2_parameters.h5'), '/dt');
pitch_axis = h5read(fullfile(data_dir,'Challenge2_2_parameters.h5'),'/pitch_axis');

%alpha_offset = 27.5; % center inputs at 0
n_modes = 20; % number of POD modes used

% Hindcast/forecast lengths
nt_hind     = 70;
nt_fore     = 130;
t_forecast = (1:nt_fore) * dt;

rng(2025); % for repeatable results

xC = pitch_axis;
yC = 0;
dx = 0.02; % for plotting airfoil location

ny = length(y);
nx = length(x);
[xx, yy] = meshgrid(x, y);

%% Helper to load variable for all cases
loadVar = @(filename, varName) load_variable_struct_from_h5(filename, varName);

%% Step (1): Read training data and test input data
fprintf('Loading training data...\n');
ux_train = loadVar(train_file, 'ux');
uy_train = loadVar(train_file, 'uy');
alpha_train = loadVar(train_file, 'alpha');
alphadot_train = loadVar(train_file, 'alphadot');

fprintf('Loading testing input data...\n');
ux_test_in  = loadVar(test_input_file, 'ux');
uy_test_in  = loadVar(test_input_file, 'uy');
alpha_test  = loadVar(test_input_file, 'alpha');
alphadot_test = loadVar(test_input_file, 'alphadot');

%% Step (2): Identify model
% Flatten training data and compute POD 
DataAllTrain = [];
aNames = fieldnames(ux_train);
for iA = 1:numel(aNames)
    fNames = fieldnames(ux_train.(aNames{iA}));
    for iF = 1:numel(fNames)
        ux = ux_train.(aNames{iA}).(fNames{iF});
        uy = uy_train.(aNames{iA}).(fNames{iF});
        nt = size(ux,3);
        DataVecs = [reshape(ux,nx*ny,nt); reshape(uy,nx*ny,nt)];
        DataAllTrain = [DataAllTrain, DataVecs];
    end
end

DataAllMean = mean(DataAllTrain,2);
DataAllTrainCentered = DataAllTrain - DataAllMean;

fprintf('Computing SVD...\n');
[Ufull, ~, ~] = svds(DataAllTrainCentered, n_modes);

DataAllTraintilde = Ufull' * DataAllTrainCentered;

%%  Compute global POD coefficient stats 
muPOD = mean(DataAllTraintilde,2); % this should be zero
sigmaPOD = std(DataAllTraintilde,0,2); % this could also be computed from singular values

%%  Compute alpha and alphadot stats from training data 
alpha_all = [];
alphadot_all = [];
for iA = 1:numel(aNames)
    fNames = fieldnames(alpha_train.(aNames{iA}));
    for iF = 1:numel(fNames)
        alpha_all = [alpha_all, alpha_train.(aNames{iA}).(fNames{iF})];
        alphadot_all = [alphadot_all, alphadot_train.(aNames{iA}).(fNames{iF})];
    end
end
muAlpha = mean(alpha_all);
sigAlpha = std(alpha_all);
muAlphadot = mean(alphadot_all);
sigAlphadot = std(alphadot_all);

%% Prepare training sequences 
DataTrainCombined = {};
indx = 0;
for iA = 1:numel(aNames)
    fNames = fieldnames(ux_train.(aNames{iA}));
    for iF = 1:numel(fNames)
        indx = indx + 1;
        ux = ux_train.(aNames{iA}).(fNames{iF});
        uy = uy_train.(aNames{iA}).(fNames{iF});
        alpha = alpha_train.(aNames{iA}).(fNames{iF});
        alphadot = alphadot_train.(aNames{iA}).(fNames{iF});
        nt = size(ux,3);
        
        DataVecs = [reshape(ux,nx*ny,nt); reshape(uy,nx*ny,nt)] - DataAllMean;
        DataTilde = Ufull' * DataVecs;
        
        DatawithKin = [(DataTilde - muPOD) ./ sigmaPOD; 
                       (alpha - muAlpha) / sigAlpha; 
                       (alphadot - muAlphadot) / sigAlphadot];
        DataTrainCombined{indx} = DatawithKin.';
    end
end

windowSize = nt_hind;
overlap = round(nt_hind/2);
dataCellTrainSeg = segmentSequences(DataTrainCombined, windowSize, overlap);

nSegsTrain = numel(dataCellTrainSeg);
Xtrain = cell(nSegsTrain,1);
Ytrain = cell(nSegsTrain,1);
for ii = 1:nSegsTrain
    Xtrain{ii} = dataCellTrainSeg{ii}(1:end-1,:);
    Ytrain{ii} = dataCellTrainSeg{ii}(2:end,1:n_modes); % only POD coeffs as outputs, as kinematics are known
end

%% Define LSTM network 
dimInputs = n_modes + 2; % POD + alpha + alphadot
dimOutputs = n_modes;

layers = [ ...
    sequenceInputLayer(dimInputs)
    lstmLayer(128)
    fullyConnectedLayer(dimOutputs)
    ];

options = trainingOptions('adam', ...
    MaxEpochs=200, ...
    MiniBatchSize=16, ...
    Shuffle='every-epoch', ...
    Plots='training-progress', ...
    Verbose=1);

%% Train LSTM
fprintf('Training LSTM...\n');
net = trainnet(Xtrain,Ytrain,layers,'mse',options);
fprintf('LSTM training complete.\n');

%% Step (3): Look at how well this model reconstructs training data
caseCount = 0;
cases2plot = [5]; % plot flow fields for these cases
ti_plot = [10,50,nt_fore]; % timesteps to plot

for iA = 1:numel(aNames)
    fNames = fieldnames(ux_train.(aNames{iA}));
    for iF = 1:numel(fNames)
        caseCount = caseCount + 1;
        
        ux = ux_train.(aNames{iA}).(fNames{iF});
        uy = uy_train.(aNames{iA}).(fNames{iF});
        alpha = alpha_train.(aNames{iA}).(fNames{iF});
        alphadot = alphadot_train.(aNames{iA}).(fNames{iF});
        
        assert(nt == size(ux,3));
        %nt_fore = size(ux_out,3);
        
        DataVecs = [reshape(ux,nx*ny,nt); reshape(uy,nx*ny,nt)] - DataAllMean;
        DataTilde = Ufull' * DataVecs;
        
        InputsScaled = [(alpha - muAlpha)/sigAlpha; (alphadot - muAlphadot)/sigAlphadot];

        xPredNorm = zeros(nt_fore,n_modes);
        xCurrNorm = [(DataTilde(:,1:nt_hind) - muPOD)./sigmaPOD; InputsScaled(:,1:nt_hind)].';
      
        net = resetState(net);
        for tind = 1:nt_fore
            xNext = predict(net,xCurrNorm);
            xPredNorm(tind,:) = xNext(end,:);
            xCurrNorm = [xCurrNorm(2:end,:); xNext(end,:), InputsScaled(:,nt_hind+tind).'];
        end
        
        xPred = xPredNorm .* sigmaPOD' + muPOD'; % map back to POD space (muPOD should be 0)
        u_forecast = Ufull * xPred.' + DataAllMean;
        ux_forecast = reshape(u_forecast(1:end/2,:), [ny,nx,nt_fore]);
        uy_forecast = reshape(u_forecast(end/2+1:end,:), [ny,nx,nt_fore]);
        ux_true = ux(:,:,nt_hind+1:nt_hind+nt_fore);
        uy_true = uy(:,:,nt_hind+1:nt_hind+nt_fore);
        ux_mean = reshape(DataAllMean(1:end/2),[ny,nx]);
        uy_mean = reshape(DataAllMean(end/2+1:end),[ny,nx]);

        diff2_dmd = (ux_true-ux_forecast).^2 + (uy_true-uy_forecast).^2;
        true_sq = (ux_true-ux_mean).^2 + (uy_true-uy_mean).^2;
        err_lstm = squeeze(mean(diff2_dmd,[1 2])).'./squeeze(mean(true_sq,[1 2])).' ;
        error_all_lstm{caseCount} = err_lstm;
     if ismember(caseCount, cases2plot)

            figure
            
            nt_plot = numel(ti_plot);
            tl_cmp = tiledlayout(4,nt_plot,'TileSpacing','compact');

            climsu = [-0.5,1.5];
            climsv = [-0.5,0.5];
            for k = 1:nt_plot
                ti = ti_plot(k);
                AoAdeg =  alpha(1,ti+nt_hind);

                nexttile(k+0*nt_plot)
                pcolor(x,y,ux_true(:,:,ti)'); shading interp, axis equal tight
                hold on
                [x1,y1] = getBodyCoordinates(AoAdeg,xC,yC,dx);
                plot(x1,y1,'k-','LineWidth',1)
                set(gca,'YTick',[],'XTick',[]); box on
                title(['$', num2str(ti),'\Delta t$'],'Interpreter','latex')
                if k == 1
                    ylabel('data','Rotation',0)
                end
                if k == nt_plot
                    cbar = colorbar('TickLabelInterpreter','latex');
                    ylabel(cbar,'$u$','interpreter','latex','Rotation',0)
                end
                clim(climsu)
                nexttile(k+1*nt_plot)
                pcolor(x,y,ux_forecast(:,:,ti)'); shading interp, axis equal tight
                hold on
                plot(x1,y1,'k-','LineWidth',1)
                set(gca,'YTick',[],'XTick',[]); box on
                clim(climsu)
                if k == 1
                    ylabel('LSTM','Rotation',0)
                end
                nexttile(k+2*nt_plot)
                pcolor(x,y,uy_true(:,:,ti)'); shading interp, axis equal tight
                hold on
                plot(x1,y1,'k-','LineWidth',1)
                set(gca,'YTick',[],'XTick',[]); box on
                clim(climsv)
                if k == 1
                    ylabel('data','Rotation',0)
                end
                if k == nt_plot
                    cbar = colorbar('TickLabelInterpreter','latex');
                    ylabel(cbar,'$v$','interpreter','latex','Rotation',0)
                end
                nexttile(k+3*nt_plot)
                pcolor(x,y,uy_forecast(:,:,ti)'); shading interp, axis equal tight
                hold on
                plot(x1,y1,'k-','LineWidth',1)
                set(gca,'YTick',[],'XTick',[]); box on
                clim(climsv)
                if k == 1
                    ylabel('LSTM','Rotation',0)
                end
                sgtitle(['Flowfield predictions, $\alpha_0 = ',aNames{iA}(2:end),'^\circ$, $f_P = 0.',fNames{iF}(4:end),'$'],'Interpreter','latex')

            end
        end
    end
end

%%
figure
hold on;
for k = 1:numel(error_all_lstm)
    plot(1:nt_fore, error_all_lstm{k} , ...
        'Color', [0.6 0.6 0.6],'linewidth',1);
end
plot(1:nt_fore, mean(cell2mat(error_all_lstm') ), ...
    'k', 'LineWidth', 1.5);
xlabel('$\Delta t$')
xlim([0,nt_fore])

ylabel('LSTM error')
title('LSTM error on training data')

%% Step (4)-(5): Generate predictions for testing data, and save to file

aNamesTest = fieldnames(ux_test_in);

% Export forecast to HDF5 for evaluation
results_file = fullfile(script_dir,'Challenge2_2_test_LSTM.h5');
if exist(results_file,'file') == 2
    delete(results_file);
end


for iA = 1:numel(aNamesTest)
    fNames = fieldnames(ux_test_in.(aNamesTest{iA}));
    for iF = 1:numel(fNames)
        

        ux_in = ux_test_in.(aNamesTest{iA}).(fNames{iF});
        uy_in = uy_test_in.(aNamesTest{iA}).(fNames{iF});
        alpha = alpha_test.(aNamesTest{iA}).(fNames{iF});
        alphadot = alphadot_test.(aNamesTest{iA}).(fNames{iF});
        
        assert(nt_hind == size(ux_in,3));
        %nt_fore = size(ux_out,3);
        
        DataVecsIn = [reshape(ux_in,nx*ny,nt_hind); reshape(uy_in,nx*ny,nt_hind)] - DataAllMean;
        DataTildeIn = Ufull' * DataVecsIn;
        
        %Inputs = [alpha; alphadot];
        InputsScaled = [(alpha - muAlpha)/sigAlpha; (alphadot - muAlphadot)/sigAlphadot];

        xPredNorm = zeros(nt_fore,n_modes);
        xCurrNorm = [(DataTildeIn - muPOD)./sigmaPOD; InputsScaled(:,1:nt_hind)].';
      
        net = resetState(net);
        for tind = 1:nt_fore
            xNext = predict(net,xCurrNorm);
            xPredNorm(tind,:) = xNext(end,:);
            xCurrNorm = [xCurrNorm(2:end,:); xNext(end,:), InputsScaled(:,nt_hind+tind).'];
        end
        
        xPred = xPredNorm .* sigmaPOD' + muPOD'; % map back to POD space (muPOD should be 0)
        u_forecast = Ufull * xPred.' + DataAllMean;
        ux_forecast = reshape(u_forecast(1:end/2,:), [ny,nx,nt_fore]);
        uy_forecast = reshape(u_forecast(end/2+1:end,:), [ny,nx,nt_fore]);

                % save results in same struct format as original data data
        % Construct dataset paths
        ux_path = ['/' aNamesTest{iA} '/' fNames{iF} '/ux'];
        uy_path = ['/' aNamesTest{iA} '/' fNames{iF} '/uy'];

        % Create datasets (HDF5 groups are implicit)
        h5create(results_file, ux_path, size(ux_forecast), 'Datatype','double');
        h5create(results_file, uy_path, size(uy_forecast), 'Datatype','double');

        % Write data
        h5write(results_file, ux_path, ux_forecast);
        h5write(results_file, uy_path, uy_forecast);
     end
end

%% save relevant model parameters
h5create(results_file,'/t',[1 nt_fore],'Datatype','double');
h5write(results_file,'/t',t_forecast);
h5create(results_file,'/x',size(x),'Datatype','double');
h5write(results_file,'/x',x);
h5create(results_file,'/y',size(y),'Datatype','double');
h5write(results_file,'/y',y);
h5writeatt(results_file,'/','method','POD-LSTM forecast');
h5writeatt(results_file,'/','nt_hind',nt_hind);
h5writeatt(results_file,'/','nt_fore',nt_fore);
h5writeatt(results_file,'/','n_modes',n_modes);
fprintf('Saved LSTM forecast to %s\n', results_file);
