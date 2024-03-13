@REM @echo off
call %CONDAPATH%\Scripts\activate.bat %CONDAPATH%
conda activate env_neural_avh
call python %*
pause
