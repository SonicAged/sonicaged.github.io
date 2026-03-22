---
title: 算法回顾-Day1
categories:
  - Daily
  - Algorithm
  - AcWing
tags:
  - 排序
  - 二分
  - 前缀和与差分
  - 双指针
  - 离散化
  - 区间合并
date: 2026-03-23 00:23:36
---

# AcWing's 算法基础课 第一讲 基础算法

此博客完全用于记录召唤死去的记忆的过程，不定时会有放空大脑的情况。

<!-- more -->

## 排序

### 快速排序

#### [快速排序](https://www.acwing.com/problem/content/787/)

```cpp
#include <iostream>
#include <algorithm>
using namespace std;

const int N = 1e5 + 10;
int n, q[N];

void quick_sort(int l, int r) {
    if(l >= r) return;

    int i = l - 1, j = r + 1, x = q[l + r >> 1];
    while(i < j) {
        do i++; while(q[i] < x);
        do j--; while(q[j] > x);
        if(i < j) swap(q[i], q[j]);
    }

    quick_sort(l, j), quick_sort(j + 1, r);
}

int main() {
    scanf("%d", &n);
    for(int i = 0; i < n; i++) scanf("%d", q + i);

    quick_sort(0, n - 1);

    for(int i = 0; i < n; i++) printf("%d ", q[i]);

    return 0;
}
```

[不赖的题解](https://www.acwing.com/solution/content/16777/)

---

#### [第 k 个数](https://www.acwing.com/problem/content/788/)
略

---

### 归并排序

#### [归并排序](https://www.acwing.com/problem/content/789/)

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int n, q[N];

void merge_sort(int l, int r) {
    if(l >= r) return;

    int mid = l + r >> 1;
    merge_sort(l, mid), merge_sort(mid + 1, r);

    int i = l, j = mid + 1, k = 0, tmp[r - l + 1];
    while(i <= mid && j <= r)
        if(q[i] < q[j]) tmp[k++] = q[i++];
        else tmp[k++] = q[j++];
    while(i <= mid) tmp[k++] = q[i++];
    while(j <= r) tmp[k++] = q[j++];

    for(int i = l; i <= r; i++) q[i] = tmp[i - l];
}

int main() {
    scanf("%d", &n);
    for(int i = 0; i < n; i++) scanf("%d", q + i);

    merge_sort(0, n - 1);

    for(int i = 0; i < n; i++) printf("%d ", q[i]);

    return 0;
}
```

[不赖的题解](https://www.acwing.com/solution/content/16778/)

---

#### [逆序对的数量](https://www.acwing.com/problem/content/790/)

- 在进行归并排序的过程中，出现了当两边都没遍历完且右边`j`对应的数比比左边`i`对应的数小时，则会产生比左边`i`到`mid`所有数对数，即`mid - i + 1`对的逆序对
- 逆序对的数量会爆`int`

```cpp
#include <iostream>
using namespace std;

typedef long long LL;
const int N = 1e5 + 10;
int n, q[N];


LL merge_sort(int l, int r) {
    if(l >= r) return 0;
    
    int mid = l + r >> 1;
    LL num = merge_sort(l, mid) + merge_sort(mid + 1, r);
    
    int i = l, j = mid + 1, k = 0, tmp[r - l + 1];
    while(i <= mid && j <= r)
        if(q[i] <= q[j]) tmp[k++] = q[i++];
        else tmp[k++] = q[j++], num += mid - i + 1;
    while(i <= mid) tmp[k++] = q[i++];
    while(j <= r) tmp[k++] = q[j++];
    
    for(int i = l, k = 0; i <= r; i++, k++) q[i] = tmp[k];
    
    return num;
}

int main() {
    scanf("%d", &n);
    for(int i = 0; i < n; i++) scanf("%d", &q[i]);

    printf("%lld", merge_sort(0, n - 1));

    return 0;
}
```

---

## 二分

### [数的范围](https://www.acwing.com/problem/content/791/)

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int n, m, q[N], k;

int main() {
    scanf("%d %d", &n, &m);
    for(int i = 0; i < n; i++) scanf("%d", &q[i]);
    for(int i = 0; i < m; i++) {
        int k;
        scanf("%d", &k);
        
        int l = 0, r = n - 1;
        while(l < r) {
            int mid = l + r >> 1;
            if(q[mid] >= k) r = mid;
            else l = mid + 1;
        }

        if(q[l] != k) printf("-1 -1\n");
        else {
            printf("%d ", l);
            l = 0, r = n - 1;
            while(l < r) {
                int mid = l + r + 1 >> 1;
                if(q[mid] <= k) l = mid;
                else r = mid - 1;
            }
            printf("%d\n", l);
        }
    }
    
    return 0;
}
```

*直接背模板吧*

```cpp
// 当需要查找左端点时
while(l < r) {
    int mid = l + r >> 1;
    if(check(mid)) r = mid;
    else l = mid + 1;
}

// 当需要查找右端点时
while(l < r) {
    int mid = l  + r + 1 >> 1;
    if(check(mid)) l = mid;
    else r = mid - 1;
}
```

---

### [数的三次方根](https://www.acwing.com/problem/content/792/)

略

---

> 摸鱼ing
> <a href="https://www.bilibili.com/video/BV1i5SeByEko/" target="_blank"><span style="display: inline-block; vertical-align: middle;"><img src="/icon/bilibili.svg" alt="bilibili" style="height: 1.1em; vertical-align: text-bottom; margin-top: 16px;"></span>开什么玩笑</a>
> <a href="https://mp.weixin.qq.com/s/lxdsq12XPmgLF9xiKb2Enw" target="_blank"><span style="display: inline-block; vertical-align: middle;"><img src="/icon/wechat.svg" alt="WeChat" style="height: 1.1em; vertical-align: text-bottom; margin-top: 16px;"></span>新闻一定要新</a>

---

## 前缀和与差分

### 前缀和

#### [前缀和](https://www.acwing.com/problem/content/797/)

```cpp
#include <iostream>
using namespace std;

const int N = 1e5 + 10;
int n, m, a, s[N], l, r;

int main() {
    cin >> n >> m;
    for (int i = 1; i <= n; i++) {
        cin >> a;
        s[i] = s[i - 1] + a;
    }

    while(m--) {
        cin >> l >> r;
        cout << s[r] - s[l - 1] << endl;
    }

    return 0;
}
```

---

#### [子矩阵的和](https://www.acwing.com/problem/content/798/)

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 1e3 + 10, Q = 2e5;
int n, m, q, s[N][N];

int main() {
    cin >> n >> m >> q;
    int a;
    for (int i = 1; i <= n; i++)
        for (int j = 1; j <= m; j++) {
            cin >> a;
            s[i][j] = s[i - 1][j] + s[i][j - 1] - s[i - 1][j - 1] + a;
        }
    
    int x1, y1, x2, y2;
    for(int i = 0; i < q; i++) {
        cin >> x1 >> y1 >> x2 >> y2;
        cout << s[x2][y2] + s[x1 - 1][y1 - 1] - s[x2][y1 - 1] - s[x1 - 1][y2] << endl;
    }
    
    return 0;
}
```

---

### 差分

#### [差分](https://www.acwing.com/problem/content/description/799/)

```cpp
#include<bits/stdc++.h>
using namespace std;

const int N = 1e5 + 10;
int n, m;
int b[N];

void insert(int l, int r, int c) {
    b[l] += c, b[r + 1] -= c;
}

int main() {
    cin >> n >> m;
    int x;
    for(int i = 1; i <= n; i++) {
        cin >> x;
        insert(i, i, x);
    }
    int l, r, c;
    for(int i = 1; i <= m; i++) {
        cin >> l >> r >> c;
        insert(l, r, c);
    }
    
    for(int i = 1; i <= n; i++) b[i] += b[i - 1];
    for(int i = 1; i <= n; i++) cout << b[i] << ' ';
    
    return 0;
}
```

---

#### [差分矩阵](https://www.acwing.com/problem/content/description/800/)

```cpp
#include<bits/stdc++.h>
using namespace std;
const int N = 1010;
int a[N][N], b[N][N], n, m, q;

void insert(int x1, int y1, int x2, int y2, int c) {
    b[x1][y1] += c, b[x2 + 1][y1] -=c, b[x1][y2 + 1] -= c, b[x2 + 1][y2 + 1] +=c;
}

int main() {
    scanf("%d%d%d", &n, &m, &q);
    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= m; j++) {
            scanf("%d", &a[i][j]);
            insert(i, j, i, j, a[i][j]);
        }
    while(q--) {
        int x1, y1, x2, y2, c;
        scanf("%d%d%d%d%d", &x1, &y1, &x2, &y2, &c);
        insert(x1, y1, x2, y2, c);
    }
    for(int i = 1; i <= n; i++)
        for(int j = 1; j <= m; j++)
            a[i][j] = b[i][j] + a[i - 1][j] + a[i][j - 1] - a[i - 1][j - 1];
    for (int i = 1; i <= n; i ++ ) {
        for (int j = 1; j <= m; j ++)
            printf("%d ", a[i][j]);
        puts("");
    }
}
```

---

## 双指针

### [最长连续不重复子序列](https://www.acwing.com/problem/content/description/801/)

- `s[i]`表示值`i`出现的次数
- `j`表示屁股，`i`表示头

```cpp
#include<iostream>
#include<algorithm>
using namespace std;

const int N = 1e5 + 10;
int n, q[N], s[N];

int main() {
    cin >> n;
    for(int i = 0; i < n; i++) cin >> q[i];
    
    int res = 0;
    for(int i = 0, j = 0; i < n; i++) {
        s[q[i]]++;
        while(s[q[i]] > 1) s[q[j++]]--;
        res = max(res, i - j + 1);
    }
    cout << res << endl;
    return 0;
}
```

---

### [数组元素的目标和](https://www.acwing.com/problem/content/802/)

```cpp
#include<iostream>
using namespace std;

const int N = 1e5 + 10;
int n, m, x;
int a[N], b[N];

int main() {
    cin >> n >> m >> x;
    for(int i = 0; i < n; i++) cin >> a[i];
    for(int i = 0; i < m; i++) cin >> b[i];

    for(int i = 0; i < n; i++) {
        int l = 0, r = m - 1;
        while(l < r) {
            int mid = l + r >> 1;
            if(b[mid] >= x - a[i]) r = mid;
            else l = mid + 1;
        }

        if(a[i] + b[l] == x) {
            cout << i << ' ' << l;
            break;
        }
    }

    return 0;
}
```

---

### [判断子序列](https://www.acwing.com/problem/content/2818/)

```cpp
#include<iostream>
using namespace std;

const int N = 1e5 + 10;
int n, m, a[N], b[N];

int main() {
    cin >> n >> m;
    for(int i = 0; i < n; i++) cin >> a[i];
    for(int i = 0; i < m; i++) cin >> b[i];

    int j = 0;
    for(int i = 0; i < m; i++) {
        if(j == n) break;
        if(a[j] == b[i]) j++;
    }

    if(j == n) cout << "Yes";
    else cout << "No";

    return 0;
}
```

---

>洗个澡捏
><img src="/img/难绷/shower.gif" alt="洗澡捏" width="400">

---

## 离散化

---

> 在此之前，还有一道关于位运算的题捏
> [二进制中1的个数](https://www.acwing.com/problem/content/803/)
> - `lowbit(x)`取出`x`中第一个`1`
>   ```cpp
>   int lowbit(int x) {
>       return x & (-x);
>   }
>   ```

---

### [区间和](https://www.acwing.com/problem/content/804/)

```cpp
#include<iostream>
#include<vector>
#include<utility>
#include<algorithm>
using namespace std;

typedef pair<int, int> PII;
const int N = 3e5 + 10;

int n, m;
int a[N], s[N];
vector<int> alls;
vector<PII> add, query;

int find(int x) {
    int l = 0, r = alls.size() - 1;
    while(l < r) {
        int mid = l + r >> 1;
        if(alls[mid] >= x) r = mid;
        else l = mid + 1;
    }
    return l;
}

int main() {
    cin >> n >> m;
    int x, c;
    while(n--) {
        cin >> x >> c;
        alls.push_back(x);
        add.push_back({x, c});
    }
    int l, r;
    while(m--) {
        cin >> l >> r;
        alls.push_back(l), alls.push_back(r);
        query.push_back({l, r});
    }

    sort(alls.begin(), alls.end());
    alls.erase(unique(alls.begin(), alls.end()), alls.end());

    for(auto item: add) a[find(item.first)] += item.second;

    for(int i = 0; i < alls.size(); i++) s[i + 1] = s[i] + a[i];

    for(auto item: query) cout << s[find(item.second) + 1] - s[find(item.first)] << endl;

    return 0;
}
```

---

## 区间合并

### [区间合并](https://www.acwing.com/problem/content/805/)

```cpp
#include<iostream>
#include<utility>
#include<vector>
#include<algorithm>
using namespace std;

typedef pair<int, int> PII;
const int N = 1e5;

int n;
vector<PII> segs;

void merge(vector<PII> &segs) {
    vector<PII> res;
    sort(segs.begin(), segs.end());
    int st = -2e9, ed = -2e9;
    for(auto seg: segs)
        if(ed < seg.first) {
            if(st != -2e9) res.push_back({st, ed});
            st = seg.first, ed = seg.second;
        }
        else ed = max(ed, seg.second);

    if(st != -2e9) res.push_back({st, ed});
    segs = res;
}

int main() {
    cin >> n;
    int l, r;
    while(n--) {
        cin >> l >> r;
        segs.push_back({l, r});
    }

    merge(segs);

    cout << segs.size();

    return 0;
}
```

---

第一讲就完了捏