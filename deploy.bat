@echo off
echo Git Push Script
echo.

echo Adding files...
git add --all

echo.
echo Committing...
git commit -m "%*"

echo.
echo Pushing to remote...
git push -u origin main

echo.
echo Done!