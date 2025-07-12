@echo off

IF "%1"=="-s" (
    echo Starting local deployment...
) 

IF "%1"=="-g" (
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

IF "%1"=="-g" (
    echo.
    echo Deploying...
    call deploy.bat %2 %3 %4 %5 %6 %7 %8 %9
)