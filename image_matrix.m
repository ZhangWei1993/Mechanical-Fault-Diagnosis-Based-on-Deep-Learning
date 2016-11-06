%this code is preparing the data for the input of the cnn using tensorflow 
%create the images
images = zeros(2400,2000);
for i = 1:10
    %addpath
    str_path = 'C:\Users\David_Zhang\Desktop\ai\bear dataset\load_1\bear_cnn_data\data_';
    str_path = strcat(str_path, num2str(i));
    load(str_path);
    filename = strcat('data_',num2str(i));
    label_orig = eval(filename);
    %save the images
    images(:,((200*(i-1)+1):200*i))= label_orig(:,(1:200));
end
images = images';
save ('images.mat', 'images');
%create the labels
labels = zeros(1,2000);
for i = 1:10
    %addpath
    str_path = 'C:\Users\David_Zhang\Desktop\ai\bear dataset\load_1\bear_cnn_data\label_';
    str_path = strcat(str_path, num2str(i));
    load(str_path);
    filename = strcat('label_',num2str(i));
    label_orig = eval(filename);
    %save the label
    labels(:,((200*(i-1)+1):200*i))= label_orig(:,(1:200));
end
labels = labels';
save ('labels.mat', 'labels');