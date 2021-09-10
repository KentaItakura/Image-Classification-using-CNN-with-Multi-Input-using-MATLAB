function prepareDigitDataset
    [XTrain,YTrain] = digitTrain4DArrayData;
    XTrainUpper=XTrain(1:14,:,:,:); % extract the upper part 
    XTrainBottom=XTrain(15:28,:,:,:);% extract the down part
    mkdir upperHalf
    mkdir bottomHalf
    for n=1:10
        mkdir(sprintf('upperHalf/%d',n-1))
        mkdir(sprintf('bottomHalf/%d',n-1))
    end

    for i=1:size(XTrainUpper,4)
        classNum=double(YTrain(i))-1; % Note that each value is added by one
        imwrite(uint8(repmat(XTrainUpper(:,:,:,i)*255,[1 1 3])),sprintf('upperHalf/%d/TrainImg%d.jpg',classNum,i))
        imwrite(uint8(repmat(XTrainBottom(:,:,:,i)*255,[1 1 3])),sprintf('bottomHalf/%d/TrainImg%d.jpg',classNum,i))
    end
end