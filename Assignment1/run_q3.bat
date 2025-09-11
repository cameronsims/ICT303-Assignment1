@ECHO off
:run
cls
rmdir /s /q "./logs/q3"
python -m q3.a1_q3

pause
C:\Users\daemc\AppData\Roaming\Python\Python313\Scripts\tensorboard.exe --logdir=logs

pause
goto run
