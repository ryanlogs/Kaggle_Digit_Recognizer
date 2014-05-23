function mat = load_csv(file,rows,columns)
% read the csv and return data in matric,
% with each line stored in a row. Skips the headers on first line

% file : name of the file
% rows : total rows to read
% columns : last column of the last row to read

	if(rows == -1 && columns == -1)
		mat = dlmread(file,',',1,0);
	else
		mat = dlmread(file,',',[1,0,1+rows,columns]);
	endif	
end