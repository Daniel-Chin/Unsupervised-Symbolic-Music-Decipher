@echo off
echo|set /p="http://localhost:6006/?pinnedCards=%5B%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22VAL_MONKEY_loss%2Fdataloader_idx_0%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22VAL_ORACLE_loss%2Fdataloader_idx_1%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22train_loss%22%7D%2C%7B%22plugin%22%3A%22scalars%22%2C%22tag%22%3A%22grad_2.0_norm_total%22%7D%5D&darkMode=true#timeseries" | clip
call %CONDAPATH%\Scripts\activate.bat env_neural_avh
tensorboard --logdir .\experiments\
