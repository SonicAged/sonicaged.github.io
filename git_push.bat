@echo off
echo Git Push Script
echo.

set /p commit_msg=Please enter commit message: 

echo.
echo Adding files...
git add .

echo.
echo Committing...
git commit -m "%commit_msg%"

echo.
echo Pushing to remote...
git push -u origin main

echo.
echo Done!
pause 