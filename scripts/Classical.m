clc; clear; close all;

%% ================== PARAMETERS ==================
Nt = 2;              % Number of transmit antennas
Nr = 2;              % Number of receive antennas
M = 4;               % QPSK (change to 16, 64, 256 later)
k = log2(M);         % Bits per symbol

numSymbols = 1e4;    % Number of symbols
snr_dB = 0:2:20;     % SNR range

BER_ZF = zeros(size(snr_dB));
BER_MMSE = zeros(size(snr_dB));

%% ================== SIMULATION ==================
for i = 1:length(snr_dB)
    
    snr = snr_dB(i);
    noise_var = 10^(-snr/10);
    
    bitErrors_ZF = 0;
    bitErrors_MMSE = 0;
    totalBits = 0;
    
    %% Run multiple blocks for averaging
    for blk = 1:100
        
        %% ----- 1. Generate bits -----
        bits = randi([0 1], numSymbols*k, 1);
        
        %% ----- 2. QAM Modulation -----
        symbols = qammod(bits, M, 'InputType','bit', 'UnitAveragePower', true);
        
        %% ----- 3. Reshape for MIMO -----
        x = reshape(symbols, Nt, []);
        
        %% ----- 4. Rayleigh Channel -----
        H = (randn(Nr,Nt) + 1j*randn(Nr,Nt))/sqrt(2);
        
        %% ----- 5. Transmit through channel -----
        y = H * x;
        
        %% ----- 6. Add noise -----
        noise = sqrt(noise_var/2) * ...
                (randn(size(y)) + 1j*randn(size(y)));
        y = y + noise;
        
        %% ================= RECEIVERS =================
        
        %% ----- ZF Detection -----
        x_zf = pinv(H) * y;
        
        %% ----- MMSE Detection -----
        x_mmse = (H'*H + noise_var*eye(Nt)) \ (H'*y);
        
        %% ----- 7. Demodulation -----
        rx_bits_zf = qamdemod(x_zf(:), M, ...
                        'OutputType','bit', ...
                        'UnitAveragePower', true);
        
        rx_bits_mmse = qamdemod(x_mmse(:), M, ...
                        'OutputType','bit', ...
                        'UnitAveragePower', true);
        
        %% ----- 8. BER Calculation -----
        bitErrors_ZF = bitErrors_ZF + sum(bits ~= rx_bits_zf);
        bitErrors_MMSE = bitErrors_MMSE + sum(bits ~= rx_bits_mmse);
        
        totalBits = totalBits + length(bits);
    end
    
    BER_ZF(i) = bitErrors_ZF / totalBits;
    BER_MMSE(i) = bitErrors_MMSE / totalBits;
    
    fprintf('SNR = %d dB | ZF BER = %.5f | MMSE BER = %.5f\n', ...
            snr, BER_ZF(i), BER_MMSE(i));
end

%% ================== PLOT ==================
figure;
semilogy(snr_dB, BER_ZF, 'r-o', 'LineWidth', 2); hold on;
semilogy(snr_dB, BER_MMSE, 'b-s', 'LineWidth', 2);
grid on;

xlabel('SNR (dB)');
ylabel('Bit Error Rate (BER)');
title('BER vs SNR for MIMO System');

legend('ZF Detector', 'MMSE Detector');