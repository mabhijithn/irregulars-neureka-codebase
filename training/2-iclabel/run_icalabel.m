function run_icalabel(startsub,stopsub,root,montage,saveroot)
%Function for run the complete icalabel to all subjects (you need to include in the path eeglab and folder with "Extra" functions
%Inputs 
%startsub,stopsub - the number of subjects for which you want to run the algorithm.
%montage - the montage for which you want to run the algorithm  (with the appropriate selection of montage startsub,stopsub you can parallelize the algorithm)
%root- The root where the data are saved - the format must follow the one given by Temple Un.
%saveroot- The folder where the results will be saved

%Example of the function paths which shall be included, root saveroot and montage formats
%cd //user//leuven//336//vsc33613//eeglab2019_1
%cd //user//leuven//336//vsc33613//Extra
%root0 = '//scratch//leuven//333//vsc33378//Datasets//Neureka_challenge';
%root1 = '//edf//dev';
%root = [root0,root1];
%montage='02_tcp_le';

eeglab
path = fullfile(root, montage);
[f,d] = getContent(path, 1);
    for ifolder = startsub:stopsub
        p = fullfile(f{ifolder}, d{ifolder});
        [f2,d2] = getContent(p, 1);
        N2 = size(d2,1);
        for isubject = 1:N2
            p = fullfile(f2{isubject}, d2{isubject});
            subjectstrname = d2{isubject};
            [f3,d3] = getContent(p, 1);
            N3 = size(d3,1); %number of sessions
            for isession = 1:N3
                pf = fullfile(f3{isession }, d3{isession });
                foldername = d3{isession};
                sessionname = strsplit(foldername,'_');
                sessionname = sessionname{1};
                recnames = findRecording(pf, subjectstrname, sessionname);
                for irec = 1:size(recnames,1)
                    fprintf('processing ifolder: %d, isubject: %d, isession: %d, irec: %d \n', ifolder, isubject, isession, irec);
                    display([pf, ' -> ', recnames.recstrname{irec}]);
                    oldedfpath = fullfile(pf, recnames.edfname{irec});
                    newedfname = [pf,'/',erase(recnames.edfname{irec},'.edf'),'_icalbl.edf'];
                    EEG=pop_biosig(oldedfpath,'importevent','off','rmeventchan','off');
                    temp=load('input.mat');
                    EEG=pop_clean_rawdata(EEG,temp.options);
                    neureka_locs=squeeze(struct2cell(readlocs('neureka.locs')));
                    eeg1=struct2cell(EEG.chanlocs);
                    for i=1:size(eeg1,2)
                        for j=1:size(neureka_locs,2)
                            if contains(eeg1{1,i},neureka_locs{3,j})
                                eeg1{3,i}=neureka_locs{1,j};
                                eeg1{4,i}=neureka_locs{2,j};
                                eeg1{5,i}=neureka_locs{8,j};
                                eeg1{6,i}=neureka_locs{9,j};
                                eeg1{7,i}=neureka_locs{10,j};
                                eeg1{8,i}=neureka_locs{4,j};
                                eeg1{9,i}=neureka_locs{5,j};
                            end
                        end
                    end
                    channels_ic=find(cellfun(@isempty,(eeg1(3,:)))==0);
                    EEG.chanlocs=cell2struct(eeg1,fieldnames(EEG.chanlocs)',1);
                    EEG=pop_runica(EEG,'chanind',channels_ic);
                    EEG=iclabel(EEG);                  
                    [indx,indy]=find(EEG.etc.ic_classification.ICLabel.classifications(:,2:6)>0.8);
                    if (size(channels_ic,2)-size(indx,1))==0
                        [indx,indy]=find(EEG.etc.ic_classification.ICLabel.classifications(:,2:6)>0.9);
                    end
                    if (size(channels_ic,2)-size(indx,1))==0
                        OUTEEG=EEG;
                    else
                        OUTEEG = pop_subcomp( EEG,indx );
                    end
                    %pfz=erase(f3{isession},'/ddn1/vol1/site_scratch/leuven/333/vsc33378/Datasets/Neureka_challenge');
                    %pfz=['/data/leuven/336/vsc33613/Neureka', pfz];
                    pfz=erase(f3{isession},root);
                    pfz=[saveroot,pfz];
                    if(~exist(pfz , 'dir'))
                        mkdir(pfz);
                    end
                    newedfname = [pfz,'/',erase(recnames.edfname{irec},'.edf'),'_icalbl.edf'];
                    pop_writeeeg(OUTEEG, newedfname, 'TYPE','EDF');
                end
            end
        end
 
    clc
    end
end


