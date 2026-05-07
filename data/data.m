clc; clear;

%% PARAMETERS
Nt = 2;
Nr = 2;
M = 4;

numSamples = 1000;

%% Precompute feature size
feat_size = 2*Nr + 2*Nr*Nt + 1;

dataset = zeros(numSamples, feat_size);

for i = 1:numSamples

    %% Generate bits
    bits = randi([0 1], Nt*log2(M), 1);

    %% Modulate
    x = qammod(bits, M, ...
        'InputType','bit', ...
        'UnitAveragePower',true);

    %% Channel
    H = (randn(Nr,Nt)+1j*randn(Nr,Nt))/sqrt(2);

    %% SNR
    snr = randi([0 20]);
    noise_var = 10^(-snr/10);

    %% Noise
    noise = sqrt(noise_var/2) * ...
        (randn(Nr,1)+1j*randn(Nr,1));

    %% Received signal
    y = H*x + noise;

    %% ===== FEATURE VECTOR (FIXED SIZE) =====
    row = [
        real(y(:));
        imag(y(:));
        real(H(:));
        imag(H(:));
        snr
    ];

    %% Convert to ROW VECTOR safely
    dataset(i,:) = row(:).';
end

%% Save dataset
writematrix(dataset,'cleaned_dataset.csv');

disp('✅ Dataset generated successfully');