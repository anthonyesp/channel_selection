function [data] = DivideTrainingTest(path,pathTL)  

    data = load(path); 

    if isempty(data) 
        error("Data is empty or path contains no data");   
    end

    f_samp = data.HDR.SampleRate; 
    SizeClassLabel= size(data.HDR.Classlabel); 

    ClassEva = load(pathTL); 

    A = isnan(data.HDR.Classlabel); 

    indexNAN = find(A==1); 

    indexClass = find(A==0) ; 

    ClassTrain  = data.HDR.Classlabel(indexClass); 
    % ClassTest   = data.HDR.Classlabel(indexNAN);  
    TrialTest =  data.HDR.TRIG(indexNAN); 
    TrialTrain =  data.HDR.TRIG(indexClass); 
    data.ClassTrain = ClassTrain ; 
    %data.ClassTest = ClassTest ;  
    data.ClassTest = ClassEva(indexNAN); 
    data.TrialTrain = TrialTrain;
    data.TrialTest = TrialTest;
end