@echo off
echo Starting local deployment...
echo.

echo Cleaning...
call hexo clean

echo.
echo Generating...
call hexo g

IF "%1"=="-s" (
    echo.
    echo Starting server...
    call hexo s
) 

IF "%1"=="-d" (
    echo.
    echo Deploying...
    call deploy.bat
)