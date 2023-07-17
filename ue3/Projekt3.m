% 1. Laden der Audiodatei
filename = './audio_file.wav';
%% Ohne native sind alles sehr kleine double werte 
%% und werden bei dem round() zu 0
%% Mit int16 macht die norm() probleme
%[x, fs] = audioread(filename, 'native');  
[x, fs] = audioread(filename);  


% 2. Quantisierung mit verschiedenen Schrittweiten
step_sizes = 1:300;
errors = zeros(size(step_sizes));
entropies = zeros(size(step_sizes));

for i = 1:length(step_sizes)
    step_size = step_sizes(i);
    % Quantisierung des Signals
    x_quantized = round(x * step_size) / step_size;
    % 3. Berechnung der Entropie
    probabilities = histcounts(x_quantized, 'Normalization', 'probability');
    entropies(i) = -sum(probabilities .* log2(probabilities + eps));
    
    % 4. Berechnung des Fehlers
    errors(i) = norm(x - x_quantized, 2);
end

% 5. Darstellung des Fehlers in Abhängigkeit der Entropie
figure;
plot(entropies, errors, 'o');
xlabel('Entropie');
ylabel('Fehler (D)');
title('Fehler in Abhängigkeit der Entropie');
grid on;