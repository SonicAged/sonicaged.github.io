@echo off

IF "%1"=="-s" (
    echo Starting local deployment...
) 

IF "%1"=="-d" (
    echo Starting global deployment...
)

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