filename = "./audio_file.wav";
[x, fs1] = audioread(filename, "native");
x = double(x);

step_sizes = 1:300;
errors = zeros(size(step_sizes));
entropies = zeros(size(step_sizes));

for i = 1:length(step_sizes)
    step_size = step_sizes(i);
    % Quantisierung des Signals
    x_hat = round(x / step_size) * step_size;

    % Histogramm erstellen
    histogram = histcounts(x_hat, 'Normalization', 'probability');

    % Entropie berechnen
    entropies(i) = -sum(histogram .* log2(histogram), 'omitnan');
    errors(i) = norm(x - x_hat, 2);
end

figure;
plot(entropies, errors, 'o');
xlabel('Entropie');
ylabel('Fehler (D)');
title('Fehler in Abhängigkeit der Entropie');
grid on;

% Normalisierung auf -1 bis 1 für sound()
x_q = (2*(double(x_q) - double(min(x_q)))/(double(max(x_q)) - double(min(x_q)))) - 1;


sound(double(x_q), fs1);


%% Aufgabe 2

[x, fs] = audioread(filename, "native");
x = double(x);

%Analyse
M=8; %Mehr Filter -> Metallischerer Klang
channelizer = dsp.Channelizer(M); % DSP Toolbox
synthesizer = dsp.ChannelSynthesizer();

y = channelizer(x);

%Quantisierung

entropies2 = zeros(M, length(step_sizes));
errors2 = zeros(M, length(step_sizes));

for i = 1:length(step_sizes)
    step_size = step_sizes(i);
    % Quantisierung des Signals
    y_hat = round(y / step_size) * step_size;
    
    %Entropie und Error pro Band
    for band=1:M
        % Histogramm erstellen
        histogram = histcounts(real(y_hat(:,band)), 'Normalization', 'probability');
    
        % Entropie berechnen
        entropies2(band, i) = -sum(histogram .* log2(histogram), 'omitnan');
        errors2(band, i) = norm(real(y(:,band)) - real(y_hat(:,band)), 2);
    end
end

figure;
for band=1:M
    plot(entropies2(band), errors2(band), 'o');
    hold on;
end
xlabel('Entropie');
ylabel('Fehler (D)');
title('Fehler in Abhängigkeit der Entropie');
grid on;

% Synthese
x_hat = double(synthesizer(y_hat));

% Normalize
x_hat = (2*(double(x_hat) - double(min(x_hat)))/(double(max(x_hat)) - double(min(x_hat)))) - 1;
sound(x_hat, fs);
    
%% Aufgabe 3
    
    

    