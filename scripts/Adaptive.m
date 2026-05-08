clc; clear; close all;

%% ================= PARAMETERS =================
Nt = 2; Nr = 2;
M = 4; k = log2(M);

numSamples = 30000;
snr_train = 0:20;

snr_dB = 0:2:20;
numSymbols = 2000;
numBlocks = 30;

input_size = 2*Nr + 2*Nr*Nt + 2*Nt;

const = qammod(0:M-1,M,'UnitAveragePower',true);

%% ================= TRAINING DATA =================
fprintf('Generating training data...\n');

X = zeros(numSamples,input_size);
Y = zeros(numSamples,2*Nt);

for i = 1:numSamples

    bits = randi([0 1],Nt*k,1);

    x = qammod(bits,M,...
        'InputType','bit',...
        'UnitAveragePower',true);

    H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);

    snr = snr_train(randi(length(snr_train)));
    noise_var = 10^(-snr/10);

    noise = sqrt(noise_var/2)*...
        (randn(Nr,1)+1j*randn(Nr,1));

    y = H*x + noise;

    %% MMSE
    x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);

    %% Residual
    residual = x - x_mmse;

    %% Features
    y_feat = [real(y); imag(y)];
    H_feat = [real(H(:)); imag(H(:))];
    mmse_feat = [real(x_mmse); imag(x_mmse)];

    X(i,:) = [y_feat; H_feat; mmse_feat]';
    Y(i,:) = [real(residual); imag(residual)]';
end

%% ================= NORMALIZATION =================
scaleX = max(abs(X),[],'all') + 1e-9;
scaleY = max(abs(Y),[],'all') + 1e-9;

X = X / scaleX;
Y = Y / scaleY;

%% ================= NETWORK =================
fprintf('Training Adaptive AI...\n');

layers = [
    featureInputLayer(input_size)

    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(256)
    batchNormalizationLayer
    reluLayer

    fullyConnectedLayer(128)
    reluLayer

    fullyConnectedLayer(2*Nt)
    regressionLayer
];

options = trainingOptions('adam',...
    'MaxEpochs',12,...
    'MiniBatchSize',256,...
    'InitialLearnRate',1e-3,...
    'Shuffle','every-epoch',...
    'Verbose',false);

net = trainNetwork(X,Y,layers,options);

disp('✅ Adaptive AI trained');

%% ================= TEST =================
BER_ZF = zeros(size(snr_dB));
BER_MMSE = zeros(size(snr_dB));
BER_AI = zeros(size(snr_dB));

fprintf('Running simulation...\n');

for s = 1:length(snr_dB)

    snr = snr_dB(s);
    noise_var = 10^(-snr/10);

    err_zf = 0;
    err_mmse = 0;
    err_ai = 0;
    totalBits = 0;

    for blk = 1:numBlocks

        %% TRANSMIT
        bits = randi([0 1],numSymbols*k,1);

        symbols = qammod(bits,M,...
            'InputType','bit',...
            'UnitAveragePower',true);

        x = reshape(symbols,Nt,[]);

        %% CHANNEL
        H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);

        noise = sqrt(noise_var/2)*...
            (randn(size(x))+1j*randn(size(x)));

        y = H*x + noise;

        %% ZF
        x_zf = pinv(H)*y;

        %% MMSE
        x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);

        %% ===== AI INPUT =====
        y_feat = [real(y); imag(y)];

        H_feat = [real(H(:)); imag(H(:))];
        H_rep = repmat(H_feat,1,size(y,2));

        mmse_feat = [real(x_mmse); imag(x_mmse)];

        input_ai = [y_feat; H_rep; mmse_feat]';
        input_ai = input_ai / scaleX;

        %% ===== PREDICT =====
        pred = predict(net,input_ai);
        pred = pred * scaleY;

        residual = pred(:,1:Nt) + ...
                   1j*pred(:,Nt+1:end);
        residual = residual.';

        x_ai = x_mmse + residual;

        %% ================= ADAPTIVE AI =================

        % SNR-based weight
        alpha_snr = 1 ./ (1 + exp((snr - 10)/2));

        % Residual confidence
        alpha_conf = exp(-abs(residual));

        % Combined adaptive weight
        alpha = alpha_snr .* alpha_conf;

        % Final adaptive output
        x_final = x_mmse + alpha .* residual;

        %% ===== HARD DECISION =====
        for n = 1:numel(x_final)
            [~,idx] = min(abs(x_final(n)-const));
            x_final(n) = const(idx);
        end

        %% DEMOD
        bits_zf = qamdemod(x_zf(:),M,...
            'OutputType','bit','UnitAveragePower',true);

        bits_mmse = qamdemod(x_mmse(:),M,...
            'OutputType','bit','UnitAveragePower',true);

        bits_ai = qamdemod(x_final(:),M,...
            'OutputType','bit','UnitAveragePower',true);

        %% SAFE LENGTH
        minLen = min([length(bits),length(bits_zf),...
                      length(bits_mmse),length(bits_ai)]);

        ref = bits(1:minLen);

        err_zf = err_zf + sum(ref ~= bits_zf(1:minLen));
        err_mmse = err_mmse + sum(ref ~= bits_mmse(1:minLen));
        err_ai = err_ai + sum(ref ~= bits_ai(1:minLen));

        totalBits = totalBits + minLen;
    end

    BER_ZF(s) = err_zf / totalBits;
    BER_MMSE(s) = err_mmse / totalBits;
    BER_AI(s) = err_ai / totalBits;

    fprintf('SNR=%2d | ZF=%.3e | MMSE=%.3e | Adaptive AI=%.3e\n',...
        snr, BER_ZF(s), BER_MMSE(s), BER_AI(s));
end

%% ================= PLOT =================
figure;

semilogy(snr_dB,BER_ZF,'r-o','LineWidth',2); hold on;
semilogy(snr_dB,BER_MMSE,'b-s','LineWidth',2);
semilogy(snr_dB,BER_AI,'k-*','LineWidth',2);

grid on;

xlabel('SNR (dB)');
ylabel('BER');

legend('ZF','MMSE','Adaptive AI');

title('Adaptive Hybrid AI Receiver');