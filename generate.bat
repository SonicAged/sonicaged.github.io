@echo off
echo Starting local deployment...
echo.

echo Cleaning...
call hexo clean

echo.
echo Generating...
call hexo g

echo.
echo Starting server...
call hexo s 