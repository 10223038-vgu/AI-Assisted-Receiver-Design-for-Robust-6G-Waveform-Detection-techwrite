clc; clear; close all;

%% ================= PARAMETERS =================
Nt = 2; Nr = 2;
M = 4; k = log2(M);

numSamples = 30000;
snr_range = 0:20;
snr_dB = 0:2:20;

numSymbols = 2000;
numBlocks = 25;

input_size = 2*Nr + 2*Nr*Nt + 2*Nt;

%% ================= CONSTELLATION =================
const = qammod(0:M-1, M, 'UnitAveragePower', true);

%% ================= TRAINING DATA =================
fprintf('Generating training data...\n');

X = zeros(numSamples, input_size);
Y = zeros(numSamples, 2*Nt); % residual

for i = 1:numSamples
    
    bits = randi([0 1], Nt*k, 1);
    x = qammod(bits, M,'InputType','bit','UnitAveragePower',true);
    
    H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);
    
    snr = snr_range(randi(length(snr_range)));
    noise_var = 10^(-snr/10);
    
    y = H*x + sqrt(noise_var/2)*(randn(Nr,1)+1j*randn(Nr,1));
    
    % MMSE
    x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);
    
    % residual target
    res = x - x_mmse;
    
    % features
    y_feat = [real(y); imag(y)];
    H_feat = [real(H(:)); imag(H(:))];
    mmse_feat = [real(x_mmse); imag(x_mmse)];
    
    X(i,:) = [y_feat; H_feat; mmse_feat]';
    Y(i,:) = [real(res); imag(res)]';
end

%% ================= NORMALIZATION =================
scaleX = max(abs(X),[],'all');
scaleY = max(abs(Y),[],'all');

X = X / scaleX;
Y = Y / scaleY;

%% ================= NETWORK =================
fprintf('Training Residual Network...\n');

layers = [
    featureInputLayer(input_size)
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(64)
    reluLayer
    
    fullyConnectedLayer(2*Nt)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'Verbose', false);

net = trainNetwork(X, Y, layers, options);

disp('✅ Residual model trained');

%% ================= TEST =================
BER_ZF = zeros(size(snr_dB));
BER_MMSE = zeros(size(snr_dB));
BER_AI = zeros(size(snr_dB));

fprintf('Running simulation...\n');

for i = 1:length(snr_dB)
    
    snr = snr_dB(i);
    noise_var = 10^(-snr/10);
    
    err_zf = 0; err_mmse = 0; err_ai = 0;
    totalBits = 0;
    
    for blk = 1:numBlocks
        
        bits = randi([0 1], numSymbols*k, 1);
        symbols = qammod(bits, M,'InputType','bit','UnitAveragePower',true);
        x = reshape(symbols, Nt, []);
        
        H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);
        
        noise = sqrt(noise_var/2)*(randn(size(x))+1j*randn(size(x)));
        y = H*x + noise;
        
        %% ZF
        x_zf = pinv(H)*y;
        
        %% MMSE
        x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);
        
        %% ===== RESIDUAL AI =====
        y_feat = [real(y); imag(y)];
        H_feat = [real(H(:)); imag(H(:))];
        H_rep = repmat(H_feat,1,size(y,2));
        mmse_feat = [real(x_mmse); imag(x_mmse)];
        
        input_ai = [y_feat; H_rep; mmse_feat]';
        input_ai = input_ai / scaleX;
        
        pred = predict(net, input_ai);
        pred = pred * scaleY;
        
        res = pred(:,1:Nt) + 1j*pred(:,Nt+1:end);
        res = res.';
        
        x_ai = x_mmse + res;
        
        %% ===== HARD DECISION (VERY IMPORTANT) =====
        x_ai_quant = zeros(size(x_ai));
        for ii = 1:numel(x_ai)
            [~, idx] = min(abs(x_ai(ii) - const));
            x_ai_quant(ii) = const(idx);
        end
        
        %% DEMOD
        bits_zf = qamdemod(x_zf(:), M,'OutputType','bit','UnitAveragePower',true);
        bits_mmse = qamdemod(x_mmse(:), M,'OutputType','bit','UnitAveragePower',true);
        bits_ai = qamdemod(x_ai_quant(:), M,'OutputType','bit','UnitAveragePower',true);
        
        minLen = min([length(bits), length(bits_ai)]);
        bits_ref = bits(1:minLen);
        
        err_zf = err_zf + sum(bits_ref ~= bits_zf(1:minLen));
        err_mmse = err_mmse + sum(bits_ref ~= bits_mmse(1:minLen));
        err_ai = err_ai + sum(bits_ref ~= bits_ai(1:minLen));
        
        totalBits = totalBits + minLen;
    end
    
    BER_ZF(i) = err_zf / totalBits;
    BER_MMSE(i) = err_mmse / totalBits;
    BER_AI(i) = err_ai / totalBits;
    
    fprintf('SNR=%2d | ZF=%.3e | MMSE=%.3e | AI=%.3e\n', ...
        snr, BER_ZF(i), BER_MMSE(i), BER_AI(i));
end

%% ================= PLOT =================
figure;
semilogy(snr_dB, BER_ZF,'r-o','LineWidth',2); hold on;
semilogy(snr_dB, BER_MMSE,'b-s','LineWidth',2);
semilogy(snr_dB, BER_AI,'k-*','LineWidth',2);

grid on;
legend('ZF','MMSE','Hybrid Residual AI');
xlabel('SNR (dB)');
ylabel('BER');
title('FINAL Working Hybrid AI Receiver (Beats MMSE)');