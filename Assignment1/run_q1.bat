@ECHO off
:run
cls
rmdir /s /q "./logs/q1"
python -m q1.a1_q1

pause
C:\Users\daemc\AppData\Roaming\Python\Python313\Scripts\tensorboard.exe --logdir=logs

pause
goto run
