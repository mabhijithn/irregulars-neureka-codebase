function run_ica(startsub,stopsub)

cd //user//leuven//336//vsc33613//eeglab2019_1
eeglab
cd //user//leuven//336//vsc33613//Extra
%% params
newroot = '//scratch//leuven//333//vsc33378//Datasets//Neureka_challenge//ICAlabel';
root0 = '//scratch//leuven//333//vsc33378//Datasets//Neureka_challenge';
root1 = '//edf//train';
root = [root0,root1];
montage = '02_tcp_le';
path = fullfile(root, montage);
[f,d] = getContent(path, 1);
N = size(d,1);


recs0=[];
    for ifolder = startsub:stopsub
        p = fullfile(f{ifolder}, d{ifolder});
        [f2,d2] = getContent(p, 1);
        N2 = size(d2,1);
        subject = [];
        for isubject = 1:N2
            p = fullfile(f2{isubject}, d2{isubject});
            subjectstrname = d2{isubject};
            subjectname = num2str(str2num(subjectstrname));
            [f3,d3] = getContent(p, 1);
            N3 = size(d3,1); %number of sessions
            
            %newsubjectfolder = fullfile(newroot,subjectname);
            %if(~exist(newsubjectfolder , 'dir'))
            %    mkdir(newsubjectfolder )
            %end
            
            for isession = 1:N3
                pf = fullfile(f3{isession }, d3{isession });
                foldername = d3{isession};
                sessionname = strsplit(foldername,'_');
                sessionname = sessionname{1};
                recnames = findRecording(pf, subjectstrname, sessionname);
                newrecCounter = 0;
                for irec = 1:size(recnames,1)
                    fprintf('processing ifolder: %d, isubject: %d, isession: %d, irec: %d \n', ifolder, isubject, isession, irec);
                    display([pf, ' -> ', recnames.recstrname{irec}]);
                    %newedfname = [subjectname, char('a' + newrecCounter)];
                    %newedfpath = fullfile(newsubjectfolder, [newedfname,'.edf']);
                    %newannpath = fullfile(newsubjectfolder, [newedfname,'.tsv']);
                    oldedfpath = fullfile(pf, recnames.edfname{irec});
                    newrecCounter = newrecCounter + 1;
                    newrecfolder = fullfile(newroot, subjectname);
                    newedfname = [subjectname,'_r', num2str(newrecCounter)];
                    newedfpath = fullfile(newrecfolder, [newedfname,'.mat']);
                    EEG=pop_biosig(oldedfpath,'importevent','off','rmeventchan','off');
                    EEG=pop_runica(EEG,'chanind',[1:21]);
                    if(~exist(newrecfolder , 'dir'))
                        mkdir(newrecfolder);
                    end
                    save(newedfpath,'EEG')
                end
            end
        end
 
    clc
    end
end


