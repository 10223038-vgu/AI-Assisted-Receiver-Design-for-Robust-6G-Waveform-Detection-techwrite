clc; clear; close all;

%% ================= PARAMETERS =================
Nt = 2;
Nr = 2;
M = 4;                 % QPSK
k = log2(M);

numSamples = 50000;    % training samples
snr_range = 0:20;      % train on multiple SNRs

snr_dB = 0:2:20;       % test SNR

%% ================= TRAINING DATA =================
input_size = 2*Nr + 2*Nr*Nt;   % y + H
output_size = 2*Nt;

Y_data = zeros(numSamples, input_size);
X_data = zeros(numSamples, output_size);

for i = 1:numSamples
    
    bits = randi([0 1], Nt*k, 1);
    
    x = qammod(bits, M, ...
        'InputType','bit','UnitAveragePower',true);
    
    H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);
    
    y = H*x;
    
    % Random SNR for robustness
    snr = snr_range(randi(length(snr_range)));
    noise_var = 10^(-snr/10);
    
    noise = sqrt(noise_var/2)*(randn(size(y))+1j*randn(size(y)));
    y = y + noise;
    
    % Prepare input (y + H)
    y_real = [real(y); imag(y)];
    H_real = [real(H(:)); imag(H(:))];
    
    Y_data(i,:) = [y_real; H_real]';
    X_data(i,:) = [real(x); imag(x)]';
end

%% ================= NORMALIZATION =================
scaleY = max(abs(Y_data),[],'all');
scaleX = max(abs(X_data),[],'all');

Y_data = Y_data / scaleY;
X_data = X_data / scaleX;

%% ================= AI MODEL =================
layers = [
    featureInputLayer(input_size)
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(64)
    reluLayer
    
    fullyConnectedLayer(output_size)
    
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 12, ...
    'MiniBatchSize', 256, ...
    'Verbose', false);

net = trainNetwork(Y_data, X_data, layers, options);

disp('✅ AI model trained');

%% ================= TEST =================
BER_ZF = zeros(size(snr_dB));
BER_MMSE = zeros(size(snr_dB));
BER_AI = zeros(size(snr_dB));

numSymbols = 2000;
numBlocks = 40;

for i = 1:length(snr_dB)
    
    snr = snr_dB(i);
    noise_var = 10^(-snr/10);
    
    err_zf = 0;
    err_mmse = 0;
    err_ai = 0;
    totalBits = 0;
    
    for blk = 1:numBlocks
        
        bits = randi([0 1], numSymbols*k, 1);
        
        symbols = qammod(bits, M, ...
            'InputType','bit','UnitAveragePower',true);
        
        x = reshape(symbols, Nt, []);
        
        H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);
        
        y = H*x;
        
        noise = sqrt(noise_var/2)*(randn(size(y))+1j*randn(size(y)));
        y = y + noise;
        
        %% ===== ZF =====
        x_zf = pinv(H)*y;
        
        %% ===== MMSE =====
        x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);
        
        %% ===== AI (FAST VECTORIZED) =====
        y_real = [real(y); imag(y)];
        H_real = [real(H(:)); imag(H(:))];
        
        % replicate H for all symbols
        H_rep = repmat(H_real,1,size(y,2));
        
        input_ai = [y_real; H_rep]';
        input_ai = input_ai / scaleY;
        
        pred = predict(net, input_ai);
        pred = pred * scaleX;
        
        x_ai = pred(:,1:Nt).' + 1j*pred(:,Nt+1:end).';
        
        %% ===== DEMOD =====
        bits_zf = qamdemod(x_zf(:), M,'OutputType','bit','UnitAveragePower',true);
        bits_mmse = qamdemod(x_mmse(:), M,'OutputType','bit','UnitAveragePower',true);
        bits_ai = qamdemod(x_ai(:), M,'OutputType','bit','UnitAveragePower',true);
        
        err_zf = err_zf + sum(bits ~= bits_zf);
        err_mmse = err_mmse + sum(bits ~= bits_mmse);
        err_ai = err_ai + sum(bits ~= bits_ai);
        
        totalBits = totalBits + length(bits);
    end
    
    BER_ZF(i) = err_zf / totalBits;
    BER_MMSE(i) = err_mmse / totalBits;
    BER_AI(i) = err_ai / totalBits;
    
    fprintf('SNR=%d | ZF=%.4f | MMSE=%.4f | AI=%.4f\n', ...
        snr, BER_ZF(i), BER_MMSE(i), BER_AI(i));
end

%% ================= PLOT =================
figure;
semilogy(snr_dB, BER_ZF, 'r-o','LineWidth',2); hold on;
semilogy(snr_dB, BER_MMSE, 'b-s','LineWidth',2);
semilogy(snr_dB, BER_AI, 'k-*','LineWidth',2);

grid on;
xlabel('SNR (dB)');
ylabel('BER');
legend('ZF','MMSE','AI');
title('Improved AI Receiver (Channel-Aware)');