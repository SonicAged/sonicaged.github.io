---
title: PEP 8
date: 2025-07-08 11:41:30
categories:
  - Python
  - CDR-pre
---

# Style Guide for Python Code

> [PEP8](https://peps.python.org/pep-0008/) 是 Python 社群共通的風格指南，一開始是 Python 之父 Guido van Rossum 自己的撰碼風格，慢慢後來演變至今，目的在於幫助開發者寫出可讀性高且風格一致的程式。許多開源計畫，例如 Django 、 OpenStack 等都是以 PEP8 為基礎再加上自己的風格建議。

这篇博客主要是为了在搭建自己的模型之前学习一下一些统一的规范是做的记录 ~~主要是目前读到的大多数论文的源码目命名没有规律~~ ，以加强之后搭建模型时代码的可读性

另外，本博客只展示本人不太熟悉的捏

## 代码布局

### 缩进

**每个缩进级别使用 4 个空格**

对于比较臭长的函数，可以使用*悬挂缩进*

```Python
# Correct:

# Aligned with opening delimiter.
foo = long_function_name(var_one, var_two,
                         var_three, var_four)

# Add 4 spaces (an extra level of indentation) to distinguish arguments from the rest.
def long_function_name(
        var_one, var_two, var_three,
        var_four):
    print(var_one)

# Hanging indents should add a level.
foo = long_function_name(
    var_one, var_two,
    var_three, var_four)
```

```Python
# Wrong:

# Arguments on first line forbidden when not using vertical alignment.
foo = long_function_name(var_one, var_two,
    var_three, var_four)

# Further indentation required as indentation is not distinguishable.
def long_function_name(
    var_one, var_two, var_three,
    var_four):
    print(var_one)
```

优先使用 _Tabs_ 进行缩进， _Tabs_ 和 _Spaces_ 不能混用

### 每行最多字符数量

**79** 个

合理使用反斜杠

```Python
with open('/path/to/some/file/you/want/to/read') as file_1, \
     open('/path/to/some/file/being/written', 'w') as file_2:
    file_2.write(file_1.read())
```

### 二元运算符之前换行

为了更好的确定该 `item` 采取的是什么运算

```Python
# Wrong:
# operators sit far away from their operands
income = (gross_wages +
          taxable_interest +
          (dividends - qualified_dividends) -
          ira_deduction -
          student_loan_interest)
```

```Python
# Correct:
# easy to match operators with operands
income = (gross_wages
          + taxable_interest
          + (dividends - qualified_dividends)
          - ira_deduction
          - student_loan_interest)
```

### 如何空行（Blank Lines）

_顶级函数_ 和 _类_ 之间空 **2** 行

_类中的函数_ 空 **1** 行

### import

- 通常每一个库 **单独一行**（也有例外）

```Python
import os
import sys

from subprocess import Popen, PIPE
```

- 按以下顺序分组，每组间空行
  1. **标准库**导入
  2. **相关第三方库**导入
  3. **特定的本地库**导入

## 注释

> Comments that contradict the code are worse than no comments.

## 命名约定

1. **类名** 用 **大驼峰**
2. **函数名** 用 **小写下划线**
3. 关于 _下划线_

   - _单下划线_ 用于占位

   ```Python
   for _ in range(10):
       print(random.randint(1, 100))
   ```

   - _单下划线_ 用于变量前表示该变量为 **弱私有** （语义上的 private）
   - _双下划线_ 用于变量前表示该变量为 **强私有** （实际上也不能调用~~实现方式是重名名~~）
