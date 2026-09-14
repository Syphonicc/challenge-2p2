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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Evaluation script 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script performs two steps: 
% 
% (1) Validates the structure and basic integrity of a submitted HDF5 
%     results file (required datasets, dimensions, etc.).
% (2) If the test output file is available locally, computes the NMSE on 
%     the mean-removed (fluctuating) fields and generates summary plots.
%
% Note: Participants typically do NOT have access to the test output data,
% which is withheld for blind evaluation. In that case, this script will
% run the file-validation checks, print a message that the test output file
% is not available, and exit without computing error metrics.
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

script_dir  = fileparts(mfilename('fullpath'));
data_dir    = fullfile(script_dir,'data');
method      = "DMD";   % "DMD" or "LSTM"

switch method
    case "DMD"
        results_file = fullfile(script_dir,'DMD','Challenge2_2_test_DMD.h5');
    case "LSTM"
        results_file = fullfile(script_dir,'LSTM','Challenge2_2_test_LSTM.h5');
    otherwise
        error('Unsupported method "%s".', method);
end

train_file = fullfile(data_dir,'Challenge2_2_train.h5');
truth_file = fullfile(data_dir,'Challenge2_2_test_output.h5');
grid_file  = fullfile(data_dir,'Challenge2_2_grid.h5');
param_file = fullfile(data_dir,'Challenge2_2_parameters.h5');

%% Load grid and parameters
x  = h5read(grid_file,'/x');
y  = h5read(grid_file,'/y');
dt = h5read(param_file,'/dt');
[xx,yy] = meshgrid(x,y);

%% Load forecast results
% Note that we have saved the results as a struct in the same format as the
% original data
% The function convert_results_struct converts this into an array with
% dimensions [nx ny nt n_seq]

fprintf('Reading results file... '); tic
if exist(results_file,'file') ~= 2
    disp('Results file not found.');
    return;
end

[ux_est, uy_est, estCases] = convert_results_struct(results_file);
fprintf('%g seconds\n', toc);

%% Load ground-truth data
if exist(truth_file,'file') ~= 2
    disp('Ground-truth file not found.');
    return;
end

fprintf('Reading test data... '); tic
[ux_true, uy_true, trueCases] = convert_results_struct(truth_file);
fprintf('%g seconds\n', toc);

% Load input data (used for computing the means that are used for scaling 
% the error quantities). Note that we only use training input data, to avoid
% any data leakage. This means that we are effectively computing error
% relative to a mean field model.
[ux_train, uy_train, trainCases] = convert_results_struct(train_file);


%% Basic compatibility checks
[nx,ny,nt_fore_est,n_seq_est] = size(ux_est);
[nx2,ny2,nt_fore_true,n_seq_true] = size(ux_true);
[~,~,nt_train,n_seq_train] = size(ux_train);

if ~isequal([nx ny],[nx2 ny2])
    disp('Spatial grid mismatch between results and truth.');
    return;
end

if nt_fore_est ~= nt_fore_true
    disp('Forecast horizon mismatch (%d vs %d).', nt_fore_est, nt_fore_true);
    return;
end

if n_seq_est ~= n_seq_true
    disp('Number of sequences mismatch (%d vs %d).', n_seq_est, n_seq_true);
    return;
end

if ~isequal(size(ux_est), size(uy_est))
    disp('Dimension mismatch between ux and uy forecasts.');
    return;
end

if ~isequal(size(ux_true), size(uy_true))
    disp('Dimension mismatch between ux and uy truth.');
    return;
end

%% Consistency check between trueCases and estCases

fprintf('\nChecking consistency between trueCases and estCases \n');

nTrue = numel(trueCases);
nEst  = numel(estCases);

if nTrue ~= nEst
    disp('Number of cases mismatch: trueCases = %d, estCases = %d.', ...
          nTrue, nEst);
    return;
end

nMismatch = 0;

for i = 1:nTrue
    aTrue = trueCases(i).aName;
    fTrue = trueCases(i).fName;

    aEst  = estCases(i).aName;
    fEst  = estCases(i).fName;

    if ~strcmp(aTrue, aEst) || ~strcmp(fTrue, fEst)
        nMismatch = nMismatch + 1;
        fprintf('Mismatch at index %d:\n', i);
        fprintf('  trueCases: a = %s, f = %s\n', aTrue, fTrue);
        fprintf('  estCases : a = %s, f = %s\n', aEst,  fEst);
    end
end

if nMismatch == 0
    fprintf('All %d cases match exactly (aName, fName).\n', nTrue);
else
    disp('Found %d mismatched case(s) between trueCases and estCases.', ...
          nMismatch);return;
end

fprintf('============================================================\n');

%%  Forecast time vector
t_est = (1:nt_fore_true) * dt;

% Calculate RMS error per forecast step
ux_true_mat = reshape(ux_true,[nx*ny, nt_fore_true, n_seq_true]);
uy_true_mat = reshape(uy_true,[nx*ny, nt_fore_true, n_seq_true]);
ux_est_mat  = reshape(ux_est, [nx*ny, nt_fore_true, n_seq_true]);
uy_est_mat  = reshape(uy_est, [nx*ny, nt_fore_true, n_seq_true]);

ux_train_mat = reshape(ux_train,[nx*ny, nt_train, n_seq_train]);
uy_train_mat = reshape(uy_train,[nx*ny, nt_train, n_seq_train]);

ux_train_mean = mean(ux_train_mat,[2,3]);
uy_train_mean = mean(uy_train_mat,[2,3]);

e_mse = zeros(nt_fore_true,n_seq_true);
% e_mean_seq = zeros(nt_fore_true,n_seq_true);
% 
% % error from mean model (equiv to fluctuation MSE)
% for i = 1:n_seq_train
%     diff_mean = (ux_true_mat(:,:,i) - ux_train_mean*ones(1,nt_fore_true)).^2 + ...
%                 (uy_true_mat(:,:,i) - uy_train_mean*ones(1,nt_fore_true)).^2;
%     e_mean_seq(:,i) = (mean(diff_mean,1)).';
% end

%% normalized mean-squared error


for i = 1:n_seq_true
    diff_sq = (ux_est_mat(:,:,i) - ux_true_mat(:,:,i)).^2 + ...
              (uy_est_mat(:,:,i) - uy_true_mat(:,:,i)).^2;
    true_sq = (ux_true_mat(:,:,i) - ux_train_mean*ones(1,nt_fore_true)).^2 ...
            + (uy_true_mat(:,:,i) - uy_train_mean*ones(1,nt_fore_true)).^2; 
    e_mse(:,i) = (mean(diff_sq,1)./mean(true_sq,1)).';
end


%% Figure: MSE error vs forecast step 
f_error = figure;
set(f_error,'Name','Forecast MSE error');
hold on;

for i = 1:n_seq_true
    plot(e_mse(:,i),'LineWidth',0.5,'Color',0.8*[1 1 1]);
end

e_mse_mean = mean(e_mse,2);
e_mse_std  = std(e_mse,0,2);

h_band = fill([1:nt_fore_true nt_fore_true:-1:1], ...
              [e_mse_mean+e_mse_std; flipud(e_mse_mean-e_mse_std)], ...
              'b','EdgeColor','none','FaceAlpha',0.1);

h_mean = plot(e_mse_mean,'b-','LineWidth',2);

box on;
xlabel('Forecast step','Interpreter','latex','FontSize',12);
ylabel('$e_\mathrm{MSE}$','Interpreter','latex','FontSize',12);
title('Forecast normalized MSE error across horizon','Interpreter','latex');
legend([h_band h_mean], {'std','mean'}, 'Interpreter','latex','Location','best');

%% Figure: sample field comparison 
f_fields  = figure;
i_display = 1;   % episode index to display

if i_display > n_seq_true
    error('Requested sequence %d exceeds available results.', i_display);
end

climsu = [-0.5,1.5];
climsv = [-0.5,0.5];

ti_plot = [1 10 50 nt_fore_true];
ti_plot = ti_plot(ti_plot <= nt_fore_true);
nt_plot = numel(ti_plot);

ux_true_disp = ux_true(:,:,:,i_display);
uy_true_disp = uy_true(:,:,:,i_display);
ux_est_disp  = ux_est(:,:,:,i_display);
uy_est_disp  = uy_est(:,:,:,i_display);

tl_cmp = tiledlayout(4,nt_plot,'TileSpacing','compact');
title(tl_cmp, sprintf('Ground truth vs forecast (%s)', method), ...
      'Interpreter','latex');

for k = 1:nt_plot
    ti = ti_plot(k);

    nexttile(k+0*nt_plot);
    pcolor(xx,yy,real(ux_true_disp(:,:,ti)));
    shading interp; axis equal tight;
    clim(climsu);
    set(gca,'XTick',[],'YTick',[]); box on;
    title(sprintf('$u^{(%d)}$',ti),'Interpreter','latex');

    nexttile(k+1*nt_plot);
    pcolor(xx,yy,real(ux_est_disp(:,:,ti)));
    shading interp; axis equal tight;
    clim(climsu);
    set(gca,'XTick',[],'YTick',[]); box on;
    title(sprintf('$\\tilde u_{%s}^{(%d)}$',method,ti),'Interpreter','latex');

    nexttile(k+2*nt_plot);
    pcolor(xx,yy,real(uy_true_disp(:,:,ti)));
    shading interp; axis equal tight;
    clim(climsv);
    set(gca,'XTick',[],'YTick',[]); box on;
    title(sprintf('$v^{(%d)}$',ti),'Interpreter','latex');

    nexttile(k+3*nt_plot);
    pcolor(xx,yy,real(uy_est_disp(:,:,ti)));
    shading interp; axis equal tight;
    clim(climsv);
    set(gca,'XTick',[],'YTick',[]); box on;
    title(sprintf('$\\tilde v_{%s}^{(%d)}$',method,ti),'Interpreter','latex');
end

colormap parula;

title(tl_cmp, sprintf('Episode %d (%s): %s,  %s', ...
    i_display, method, ...
    trueCases(i_display).aName, ...
    trueCases(i_display).fName), ...
    'Interpreter','latex');


fprintf('Challenge 2.2 evaluation complete.\n');
%%


%% Required function for loading data
function [ux, uy, caseInfo] = convert_results_struct(filename)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Load Challenge 2.2 hierarchical HDF5 results or truth files and assemble
% them into flat arrays compatible with evaluation script.
%
% Expected HDF5 structure:
%   /aXX/fYY/ux   [ny nx nt]
%   /aXX/fYY/uy   [ny nx nt]
%
% Outputs:
%   ux       [nx ny nt n_seq]
%   uy       [nx ny nt n_seq]
%   caseInfo struct array with fields:
%              .aName   (e.g. 'a25')
%              .fName   (e.g. 'f0p05')
%              .path    (full HDF5 group path)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

if exist(filename,'file') ~= 2
    disp('HDF5 file not found: %s', filename);
    return;
end

info = h5info(filename);
aGroups = info.Groups;

if isempty(aGroups)
    error('No top-level groups found in file: %s', filename);
end

% First pass: count sequences and infer dimensions
n_seq = 0;

for iA = 1:numel(aGroups)
    fGroups = aGroups(iA).Groups;
    for iF = 1:numel(fGroups)
        n_seq = n_seq + 1;

        if n_seq == 1
            % Read one dataset to get dimensions
            ux_tmp = h5read(filename, [fGroups(iF).Name '/ux']);
            if ndims(ux_tmp) ~= 3
                error('Dataset %s/ux must be 3-D [ny nx nt].', fGroups(iF).Name);
            end
            [ny,nx,nt] = size(ux_tmp);
        end
    end
end

if n_seq == 0
    error('No (a,f) cases found in file: %s', filename);
end

% Preallocate output arrays
ux = zeros(nx,ny,nt,n_seq);
uy = zeros(nx,ny,nt,n_seq);

caseInfo = struct('aName',{},'fName',{},'path',{});
caseInfo(n_seq).aName = []; % preallocate struct array

% Second pass: load data
seq = 0;
for iA = 1:numel(aGroups)
    aName = aGroups(iA).Name;
    [~,aLabel] = fileparts(aName);

    fGroups = aGroups(iA).Groups;
    for iF = 1:numel(fGroups)
        seq = seq + 1;

        fName = fGroups(iF).Name;
        [~,fLabel] = fileparts(fName);

        ux_tmp = h5read(filename, [fName '/ux']);
        uy_tmp = h5read(filename, [fName '/uy']);

        % Dimension checks
        if ~isequal(size(ux_tmp), [ny nx nt])
            error('Inconsistent ux dimensions in %s.', fName);
        end
        if ~isequal(size(uy_tmp), [ny nx nt])
            error('Inconsistent uy dimensions in %s.', fName);
        end

        % Permute to [nx ny nt]
        ux(:,:,:,seq) = permute(ux_tmp,[2 1 3]);
        uy(:,:,:,seq) = permute(uy_tmp,[2 1 3]);

        % Store metadata
        caseInfo(seq).aName = aLabel;
        caseInfo(seq).fName = fLabel;
        caseInfo(seq).path  = fName;
    end
end

end
