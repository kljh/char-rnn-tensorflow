@echo off

echo List files
dir /b /s ..\..\..\vision\*.js  > filelist.txt
type filelist.txt

echo Concatenate files
echo. > input.js
for /f %%f in ( filelist.txt ) do type %%f >> input.js

echo Done BUT STILL CONTAINS UTF-8 / OTHER ENCODING CHARACTERS
