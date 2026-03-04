# 070. Climbing Stairs（爬楼梯）

## 题目描述

假设你正在爬楼梯。需要 `n` 阶你才能到达楼顶。每次你可以爬 `1` 或 `2` 个台阶。问有多少种不同的方法可以爬到楼顶？

## 解题思路

经典 DP：到达第 i 阶的方法数 = 到达第 i-1 阶的方法数 + 到达第 i-2 阶的方法数（斐波那契数列）。

## 代码实现

```python
def climbStairs(n):
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2  # 分别代表 dp[i-2] 和 dp[i-1]
    
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2 = prev1
        prev1 = curr
    
    return prev1
```

## 复杂度分析

- **时间复杂度**：O(n) - 遍历一次
- **空间复杂度**：O(1) - 只存储前两个状态

## 关键点

- 本质是斐波那契数列
- 可用滚动数组优化空间至 O(1)
- 注意边界条件 n = 1, 2
