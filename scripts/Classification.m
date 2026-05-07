clc; clear; close all;

%% ================= PARAMETERS =================
Nt = 2; Nr = 2;
M = 4; k = log2(M);

numSamples = 30000;     % training samples (balanced speed/accuracy)
snr_range = 0:20;
snr_dB = 0:2:20;

numSymbols = 2000;
numBlocks = 25;

input_size = 2*Nr + 2*Nr*Nt + 2*Nt + Nt; % + antenna indicator

%% ================= CONSTELLATION =================
const = qammod(0:M-1, M, 'UnitAveragePower', true);

%% ================= TRAINING DATA =================
fprintf('Generating training data...\n');

X_data = zeros(numSamples*Nt, input_size);
Y_labels = zeros(numSamples*Nt,1);

idx = 1;

for i = 1:numSamples
    
    bits = randi([0 1], Nt*k, 1);
    x = qammod(bits, M,'InputType','bit','UnitAveragePower',true);
    
    H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);
    
    snr = snr_range(randi(length(snr_range)));
    noise_var = 10^(-snr/10);
    
    y = H*x + sqrt(noise_var/2)*(randn(Nr,1)+1j*randn(Nr,1));
    
    % MMSE estimate
    x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);
    
    % Features
    y_feat = [real(y); imag(y)];
    H_feat = [real(H(:)); imag(H(:))];
    mmse_feat = [real(x_mmse); imag(x_mmse)];
    
    base_input = [y_feat; H_feat; mmse_feat];
    
    for t = 1:Nt
        
        antenna_feat = zeros(Nt,1);
        antenna_feat(t) = 1;
        
        X_data(idx,:) = [base_input; antenna_feat]';
        
        [~,label] = min(abs(x(t) - const));
        Y_labels(idx) = label;
        
        idx = idx + 1;
    end
end

Y_labels = categorical(Y_labels);

%% ================= NORMALIZATION =================
scaleX = max(abs(X_data),[],'all');
X_data = X_data / scaleX;

%% ================= NETWORK =================
fprintf('Training AI model...\n');

layers = [
    featureInputLayer(input_size)
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(64)
    reluLayer
    
    fullyConnectedLayer(M)
    softmaxLayer
    classificationLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 10, ...
    'MiniBatchSize', 256, ...
    'Verbose', false);

net = trainNetwork(X_data, Y_labels, layers, options);

disp('✅ AI training complete');

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
        
        %% ===== ZF =====
        x_zf = pinv(H)*y;
        
        %% ===== MMSE =====
        x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);
        
        %% ===== AI =====
        x_ai = zeros(Nt, size(y,2));
        
        y_feat = [real(y); imag(y)];
        H_feat = [real(H(:)); imag(H(:))];
        H_rep = repmat(H_feat,1,size(y,2));
        mmse_feat = [real(x_mmse); imag(x_mmse)];
        
        for t = 1:Nt
            
            antenna_feat = zeros(Nt,1);
            antenna_feat(t) = 1;
            
            input_ai = [y_feat; H_rep; mmse_feat; repmat(antenna_feat,1,size(y,2))]';
            input_ai = input_ai / scaleX;
            
            preds = classify(net, input_ai);
            labels = double(preds);
            
            x_ai(t,:) = const(labels).';
        end
        
        %% ===== DEMOD =====
        bits_zf = qamdemod(x_zf(:), M,'OutputType','bit','UnitAveragePower',true);
        bits_mmse = qamdemod(x_mmse(:), M,'OutputType','bit','UnitAveragePower',true);
        bits_ai = qamdemod(x_ai(:), M,'OutputType','bit','UnitAveragePower',true);
        
        % Safe comparison
        minLen = min([length(bits), length(bits_zf), length(bits_mmse), length(bits_ai)]);
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
legend('ZF','MMSE','AI (Classification)');
xlabel('SNR (dB)');
ylabel('BER');
title('Final Clean AI Receiver (Stable & Bug-Free)');