function [recs] = findRecording(root, subjectstrname, sessionname)
ext = 'edf';
[f,d] = getContent(root, 0);
N = size(d,1);
recnames=[];
r=[];
for i = 1:N
    ss = strsplit(d{i},{'_','.'});
    if(length(ss)~= 4)
        continue
    end
    if(strcmp(ss{1}, subjectstrname))
        if(strcmp(ss{2}, sessionname))
            if(strcmp(ss{4}, ext))
                r.recstrname = ss(3);
                r.recnum = str2num(r.recstrname{1}(2:end));
                r.edfname = d(i);
                r.lblname = {[subjectstrname,'_',sessionname,'_',r.recstrname{1},'.lbl']};
                recnames=[recnames; r];
            end
        end
    end
end
recs = struct2table(recnames);
recs = sortrows(recs, 'recnum');
end