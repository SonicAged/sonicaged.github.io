#!/bin/bash

if [ "$1" == "-l" ]; then
    echo "Starting local deployment..."
fi

if [ "$1" == "-g" ]; then
    echo "Starting global deployment..."
fi

echo
echo "Cleaning..."
hexo clean

echo
echo "Generating..."
hexo g

if [ "$1" == "-l" ]; then
    echo
    echo "Starting server..."
    hexo s --draft
fi

if [ "$1" == "-g" ]; then
    echo
    echo "Deploying..."
    # shift 的作用是移除第一个参数 -g，将其余参数传递给下一个脚本
    shift
    # 假设你的第一个脚本保存为 push.sh
    ./deploy.sh "$@"
fi