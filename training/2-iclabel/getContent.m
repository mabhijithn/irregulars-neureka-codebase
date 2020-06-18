function [dirs, fnames] = getContent(root, isdir)
d = struct2table(dir(root));
d(strcmp(d.name, '.'),:)=[];
d(strcmp(d.name, '..'),:)=[];
if(~isdir)
    d(d.isdir,:) = [];
end

fnames = d.name;
dirs= d.folder;
end