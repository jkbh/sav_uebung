filename = "./audio_file.wav";
[x, fs1] = audioread(filename, "native");
[y, fs2] = audioread(filename);

step_sizes = 1:300;
errors = zeros(size(step_sizes));
entropies = zeros(size(step_sizes));

for i = 1:length(step_sizes)
    step_size = step_sizes(i);
    % Quantisierung des Signals
    x_q = round(x / step_size) * step_size;

    % Histogramm erstellen
    histogram = histcounts(x_q, 'Normalization', 'probability');

    % Entropie berechnen
    entropies(i) = -sum(histogram .* log2(histogram), 'omitnan');
    errors(i) = norm(double(x) - double(x_q), 2);
end

figure;
plot(entropies, errors, 'o');
xlabel('Entropie');
ylabel('Fehler (D)');
title('Fehler in Abhängigkeit der Entropie');
grid on;

% Normalisierung auf -1 bis 1 für sound()
normalized_vec = (2*(double(x_q) - double(min(x_q)))/(double(max(x_q)) - double(min(x_q)))) - 1;


sound(normalized_vec, fs1);
