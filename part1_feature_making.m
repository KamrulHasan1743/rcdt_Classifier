clc
close all
clear all
kk=dir('grayscale_frames/*.png');
addpath([pwd '/grayscale_frames'])
cnt=1;
for i=1:length(kk)
    if mod(i,10)==0
        [i ]
    end
    y=kk(i).name;
    img=imread(y);
    figure,imshow(img,[min(min(img)) max(max(img))])
    size(img)
    if sum(sum(img))
        feat(cnt,:)=RCDT_features(img);
        labl(cnt)=y(1)-96;
        cnt=cnt+1;
    end
end
return
size(feat)
save 'feat_bin.mat' feat labl