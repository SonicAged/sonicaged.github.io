#!/bin/bash

# 脚本在遇到任何错误时立即退出
set -e

# 检查是否提供了提交信息
if [ -z "$1" ]; then
  echo "错误：请提供一个提交信息作为参数。"
  echo "用法: ./git-push.sh \"你的提交信息\""
  exit 1
fi

echo "Git Push 脚本"
echo

echo "正在添加所有文件..."
git add --all

echo
echo "正在提交..."
# "$*" 会将所有命令行参数作为一个单一的字符串
git commit -m "$*"

echo
echo "正在推送到远程仓库..."
git push -u origin main

echo
echo "完成！"