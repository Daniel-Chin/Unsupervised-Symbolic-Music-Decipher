@echo off
call %CONDAPATH%\Scripts\activate.bat env_neural_avh
tensorboard --logdir .\experiments\
