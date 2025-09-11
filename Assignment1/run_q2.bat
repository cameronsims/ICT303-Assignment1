@ECHO off
:run
cls
rmdir /s /q "./logs/q2"
python -m q2.a1_q2

pause
C:\Users\daemc\AppData\Roaming\Python\Python313\Scripts\tensorboard.exe --logdir=logs

pause
goto run
