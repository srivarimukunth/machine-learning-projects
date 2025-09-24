%% run_major_project_spo2_qrs_enhanced.m
% End-to-end BMET5934 pipeline (SpO2 + QRS-HR/HRV), memory-safe
% - Hold-out by records (prints Sensitivity, PPV, F1) with post-processing
% - Logistic classifier (fitclinear)
% - Submission: ProjectTestAnnotationsGroup<id>Submission<k>.mat (ONLY Class)

%% ============================ CONFIG ===============================
GROUP_NUM      = 5;          % <-- set your group number
SUBMISSION_NUM = 2;          % <-- 1..5
VAL_HOLDOUT    = 0.20;       % validation fraction by records
THRESH_GRID    = 0.30:0.02:0.70;   % scan thresholds to maximize F1
SMOOTH_WIN     = 11;         % seconds (odd) median filter window
MIN_A_RUN      = 10;         % minimum apnoea duration (seconds)
RANDOM_SEED    = 5934;
% ====================================================================

rng(RANDOM_SEED);

%% ==================== Load TRAIN (SpO2 + QRS only) ==================
trainFile = 'ProjectTrainData.mat';
if ~isfile(trainFile), error('Cannot find %s', trainFile); end
S = load(trainFile, 'SpO2','Class','QRS','SR_SpO2','SR_ECG');
fs_spo2 = S.SR_SpO2; fs_ecg = S.SR_ECG;
n_records = numel(S.SpO2);
fprintf('Training on %d records (SpO2 @ %g Hz; ECG @ %g Hz for QRS)\n', n_records, fs_spo2, fs_ecg);

%% ======= Preallocate (seconds across all training records) ==========
total_sec = 0;
for i = 1:n_records, total_sec = total_sec + numel(S.SpO2{i}); end
D = 16;                                % feature count (see maker below)
X = zeros(total_sec, D, 'single');     % features (single -> memory safe)
Y = false(total_sec, 1);               % labels
RECID = zeros(total_sec, 1, 'uint16'); % record id per row

%% ======= Compute features per record and fill matrices ==============
row = 1;
for i = 1:n_records
    Fi = make_features_spo2_qrs_hrv(S.SpO2{i}, S.QRS{i}, fs_ecg);  % [sec x D]
    Li = (S.Class{i}=='A');
    n  = min(size(Fi,1), numel(Li));
    X(row:row+n-1, :) = Fi(1:n, :);
    Y(row:row+n-1)    = Li(1:n);
    RECID(row:row+n-1)= i;
    row = row + n;
    if mod(i,10)==0 || i==n_records
        fprintf('  processed %d/%d training records\n', i, n_records);
    end
end
clear Fi Li

%% ======= Hold-out split by records (no leakage) =====================
cvRecs   = cvpartition(n_records, 'Holdout', VAL_HOLDOUT);
valRecs  = find(test(cvRecs));
isValRow = ismember(RECID, valRecs);

Xtr = X(~isValRow,:);  Ytr = Y(~isValRow);
Xva = X(isValRow,:);   Yva = Y(isValRow);
RECID_va = RECID(isValRow);

%% ======= Scale using TRAIN stats; reuse on test =====================
mu = mean(double(Xtr), 1);
sg = std(double(Xtr), 0, 1) + 1e-8;
Xtr = single((double(Xtr) - mu) ./ sg);  Xtr(~isfinite(Xtr)) = 0;
Xva = single((double(Xva) - mu) ./ sg);  Xva(~isfinite(Xva)) = 0;

%% ======= Train fast logistic classifier (class-weighted) ============
pos = mean(Ytr);
w1  = 0.5 / max(pos,1e-6);
w0  = 0.5 / max(1-pos,1e-6);
Wtr = double(Ytr).*w1 + double(~Ytr).*w0;

mdl = fitclinear(double(Xtr), Ytr, ...
    'Learner','logistic','Solver','lbfgs', ...
    'ClassNames',[false true], 'Weights', Wtr);

%% ======= Validation performance with post-processing =================
[~, scoVA] = predict(mdl, double(Xva));
p = scoVA(:,2);

best = struct('t',0.5,'F1',-inf,'P',NaN,'R',NaN);
for t = THRESH_GRID
    [F1,P,R] = score_with_postproc(p, Yva, RECID_va, valRecs, t, SMOOTH_WIN, MIN_A_RUN);
    if F1 > best.F1, best = struct('t',t,'F1',F1,'P',P,'R',R); end
end

fprintf('\nHOLD-OUT (with smoothing %ds & min-run %ds)\n', SMOOTH_WIN, MIN_A_RUN);
fprintf('  Threshold                = %.2f\n', best.t);
fprintf('  Sensitivity (Recall)     = %.4f\n', best.R);
fprintf('  Positive Predictivity    = %.4f\n', best.P);
fprintf('  F1-score                 = %.4f\n\n', best.F1);

% Free memory before test stage
clear X Y Xtr Xva Ytr Yva scoVA p Wtr isValRow

%% ===================== Predict TEST and build Class ==================
testDataFile = 'ProjectTestData.mat';
testAnnoFile = 'ProjectTestAnnotations.mat';
%if ~isfile(testDataFile) || !isfile(testAnnoFile)
   % error('Missing test files: %s or %s not found.', testDataFile, testAnnoFile);
%end

T  = load(testDataFile, 'SpO2','QRS','SR_SpO2','SR_ECG');
A  = load(testAnnoFile, 'Class');

if T.SR_SpO2 ~= fs_spo2 || T.SR_ECG ~= fs_ecg
    warning('Train/test rates differ (train SpO2=%g/ECG=%g, test SpO2=%g/ECG=%g). Proceeding...', ...
        fs_spo2, fs_ecg, T.SR_SpO2, T.SR_ECG);
end

Class = A.Class;
num_test = numel(T.SpO2);
for i = 1:num_test
    Fi = make_features_spo2_qrs_hrv(T.SpO2{i}, T.QRS{i}, fs_ecg);
    Xi = single((double(Fi) - mu) ./ sg);  Xi(~isfinite(Xi)) = 0;

    [~, sc] = predict(mdl, double(Xi));
    pred = sc(:,2) >= best.t;

    % Apply the SAME post-processing as validation
    if SMOOTH_WIN > 1 && mod(SMOOTH_WIN,2)==1
        pred = medfilt1(double(pred), SMOOTH_WIN) >= 0.5;
    end
    if MIN_A_RUN > 1
        pred = enforce_min_run(pred, MIN_A_RUN);
    end

    L = numel(Class{i});
    if numel(pred) < L, pred(end+1:L) = false; end
    if numel(pred) > L, pred = pred(1:L); end

    out = repmat('N', L, 1);  out(pred) = 'A';
    Class{i} = out;

    if mod(i,10)==0 || i==num_test
        fprintf('  predicted %d/%d test records\n', i, num_test);
    end
end

outName = sprintf('ProjectTestAnnotationsGroup%dSubmission%d.mat', GROUP_NUM, SUBMISSION_NUM);
save(outName, 'Class');
fprintf('Submission saved as %s (only `Class`)\n', outName);

%% ============================= Helpers ==============================
function F = make_features_spo2_qrs_hrv(spo2, qrs, fs_ecg)
% Per-second features (single), D=16:
% HR_mean10, HR_std10, RR_mean10, RR_std10, HR_slope5, HR_var60, RR_var60,
% Sp_mean10, Sp_min10, Sp_std10, Sp_drop10, Sp_delta10, Sp_below90,
% Sp_slope5, ODI30, Sp_below90_60
Sp = double(spo2(:));
N  = numel(Sp);
sec = (1:N)';

% ---- HR & RR per second from QRS ----
rrs = diff(qrs);
if isempty(rrs)
    hr_sec = zeros(N,1);
    rr_sec = nan(N,1);
else
    rr = rrs / fs_ecg;           % s
    hr = 60 ./ rr;               % bpm
    bt = qrs(2:end) / fs_ecg;    % beat times (s)
    hr_sec = interp1(bt, hr, sec, 'previous','extrap');
    rr_sec = interp1(bt, rr, sec, 'previous','extrap');
end

w10 = 10;
HR_mean10 = movmean(hr_sec, [w10-1 0], 'omitnan');
HR_std10  = movstd( hr_sec, [w10-1 0], 'omitnan');
RR_mean10 = movmean(rr_sec, [w10-1 0], 'omitnan');
RR_std10  = movstd( rr_sec, [w10-1 0], 'omitnan');
HR_slope5 = [0; diff(movmean(hr_sec, [4 0], 'omitnan'))];
HR_var60  = movvar(hr_sec, [59 0], 'omitnan');
RR_var60  = movvar(rr_sec, [59 0], 'omitnan');

% ---- SpO2 features ----
mean10   = movmean(Sp, [w10-1 0], 'omitnan');
min10    = movmin( Sp, [w10-1 0], 'omitnan');
std10    = movstd( Sp, [w10-1 0], 'omitnan');
drop10   = Sp - min10;
delta10  = Sp - [nan(w10,1); Sp(1:end-w10)];  delta10(1:w10)=0;
below90  = double(Sp < 90);
slope5   = [0; diff(movmean(Sp, [4 0], 'omitnan'))];
desat4   = (drop10 <= -4);
ODI30    = movsum(desat4, [29 0], 'omitnan');
below90_60 = movsum(Sp < 90, [59 0], 'omitnan');  % seconds <90 in last 60 s

F = single([HR_mean10, HR_std10, RR_mean10, RR_std10, HR_slope5, HR_var60, RR_var60, ...
            mean10, min10, std10, drop10, delta10, below90, slope5, ODI30, below90_60]);
F(~isfinite(F)) = 0;
end

function [F1, P, R] = score_with_postproc(p, y, recid, rec_list, t, smooth_win, min_run)
% Apply threshold + post-proc per validation record, then aggregate metrics
TP=0; FP=0; FN=0;
for r = reshape(rec_list,1,[])
    idx = (recid == r);
    if ~any(idx), continue; end
    pred = p(idx) >= t;
    if smooth_win > 1 && mod(smooth_win,2)==1
        pred = medfilt1(double(pred), smooth_win) >= 0.5;
    end
    if min_run > 1
        pred = enforce_min_run(pred, min_run);
    end
    yt = y(idx);
    TP = TP + sum(pred & yt);
    FP = FP + sum(pred & ~yt);
    FN = FN + sum(~pred & yt);
end
P = TP / (TP + FP + eps);
R = TP / (TP + FN + eps);
F1 = 2*P*R / (P + R + eps);
end

function y = enforce_min_run(x, minlen)
x = x(:)' > 0; y = x;
d = diff([0, x, 0]); s = find(d==1); e = find(d==-1)-1;
for k = 1:numel(s)
    if (e(k)-s(k)+1) < minlen, y(s(k):e(k)) = 0; end
end
y = y(:) > 0;
end
