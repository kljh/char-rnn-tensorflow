@echo off

echo List files
dir /b /s ..\..\vision\*.js  > alljs.txt
::type alljs.txt

echo Concatenate files
echo. > input.js
for /f %%f in ( alljs.txt ) do type %%f >> input.js

echo Done BUT STILL CONTAINS UTF-8 / OTHER ENCODING CHARACTERS
