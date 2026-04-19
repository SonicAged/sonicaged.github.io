#!/bin/bash

# 脚本在遇到任何错误时立即退出
set -e

# 检查是否提供了提交信息
if [ -z "$1" ]; then
  echo "ERROR: Please provide a commit message."
  echo "Usage: ./deploy.sh \"Your commit message\""
  exit 1
fi

echo "Git Push Script"
echo

echo "Adding all files..."
git add --all

echo
echo "Committing..."
# "$*" 会将所有命令行参数作为一个单一的字符串
git commit -m "$*"

echo
echo "Pushing to remote repository..."
git push -u origin main

echo
echo "Done!"