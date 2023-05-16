function [labels] = InstrumentClassifier(train_data, train_labels, test_data)
Mdl = TreeBagger(250,train_data, train_labels); 
labels= predict(Mdl, test_data);
labels = str2double(labels);
end