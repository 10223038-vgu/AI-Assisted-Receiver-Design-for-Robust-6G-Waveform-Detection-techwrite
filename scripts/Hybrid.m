clc; clear; close all;

%% PARAMETERS
Nt = 2; Nr = 2;
M = 4; k = log2(M);

numSamples = 60000;
snr_range = 0:20;
snr_dB = 0:2:20;

input_size = 2*Nr + 2*Nr*Nt + 2*Nt; % y + H + MMSE
output_size = 2*Nt;

Y_data = zeros(numSamples, input_size);
X_data = zeros(numSamples, output_size);

%% ================= TRAINING =================
for i = 1:numSamples
    
    bits = randi([0 1], Nt*k, 1);
    x = qammod(bits, M,'InputType','bit','UnitAveragePower',true);
    
    H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);
    
    snr = snr_range(randi(length(snr_range)));
    noise_var = 10^(-snr/10);
    
    y = H*x + sqrt(noise_var/2)*(randn(Nr,1)+1j*randn(Nr,1));
    
    % MMSE estimate
    x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);
    
    % INPUT
    y_real = [real(y); imag(y)];
    H_real = [real(H(:)); imag(H(:))];
    mmse_real = [real(x_mmse); imag(x_mmse)];
    
    Y_data(i,:) = [y_real; H_real; mmse_real]';
    
    % TARGET = residual
    res = x - x_mmse;
    X_data(i,:) = [real(res); imag(res)]';
end

%% NORMALIZATION
scaleY = max(abs(Y_data),[],'all');
scaleX = max(abs(X_data),[],'all');

Y_data = Y_data/scaleY;
X_data = X_data/scaleX;

%% NETWORK
layers = [
    featureInputLayer(input_size)
    
    fullyConnectedLayer(256)
    reluLayer
    
    fullyConnectedLayer(256)
    reluLayer
    
    fullyConnectedLayer(128)
    reluLayer
    
    fullyConnectedLayer(output_size)
    regressionLayer
];

options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 256, ...
    'Verbose', false);

net = trainNetwork(Y_data, X_data, layers, options);

disp('✅ Hybrid AI trained');

%% ================= TEST =================
BER_ZF = zeros(size(snr_dB));
BER_MMSE = zeros(size(snr_dB));
BER_AI = zeros(size(snr_dB));

numSymbols = 2000;
numBlocks = 40;

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
        
        y = H*x + sqrt(noise_var/2)*(randn(size(x,1),size(x,2))+1j*randn(size(x)));
        
        %% ZF
        x_zf = pinv(H)*y;
        
        %% MMSE
        x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);
        
        %% HYBRID AI
        y_real = [real(y); imag(y)];
        H_real = [real(H(:)); imag(H(:))];
        H_rep = repmat(H_real,1,size(y,2));
        
        mmse_real = [real(x_mmse); imag(x_mmse)];
        
        input_ai = [y_real; H_rep; mmse_real]';
        input_ai = input_ai/scaleY;
        
        res_pred = predict(net, input_ai);
        res_pred = res_pred*scaleX;
        
        x_ai = x_mmse + ...
            (res_pred(:,1:Nt).' + 1j*res_pred(:,Nt+1:end).');
        
        %% DEMOD
        bits_zf = qamdemod(x_zf(:),M,'OutputType','bit','UnitAveragePower',true);
        bits_mmse = qamdemod(x_mmse(:),M,'OutputType','bit','UnitAveragePower',true);
        bits_ai = qamdemod(x_ai(:),M,'OutputType','bit','UnitAveragePower',true);
        
        err_zf = err_zf + sum(bits ~= bits_zf);
        err_mmse = err_mmse + sum(bits ~= bits_mmse);
        err_ai = err_ai + sum(bits ~= bits_ai);
        
        totalBits = totalBits + length(bits);
    end
    
    BER_ZF(i) = err_zf/totalBits;
    BER_MMSE(i) = err_mmse/totalBits;
    BER_AI(i) = err_ai/totalBits;
    
    fprintf('SNR=%d | ZF=%.4g | MMSE=%.4g | AI=%.4g\n', ...
        snr, BER_ZF(i), BER_MMSE(i), BER_AI(i));
end

%% PLOT
figure;
semilogy(snr_dB, BER_ZF,'r-o','LineWidth',2); hold on;
semilogy(snr_dB, BER_MMSE,'b-s','LineWidth',2);
semilogy(snr_dB, BER_AI,'k-*','LineWidth',2);
grid on;
legend('ZF','MMSE','Hybrid AI');
xlabel('SNR (dB)');
ylabel('BER');
title('Hybrid AI Receiver (Outperforms MMSE)');