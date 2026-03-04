# 001. Two Sum（两数之和）

## 题目描述

给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** 的那 **两个** 整数，并返回它们的数组下标。

## 解题思路

使用哈希表存储已遍历的数字及其索引。遍历数组时，检查 `target - nums[i]` 是否在哈希表中。

## 代码实现

```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []
```

## 复杂度分析

- **时间复杂度**：O(n) - 只需遍历数组一次
- **空间复杂度**：O(n) - 哈希表存储 n 个元素

## 关键点

- 哈希表可实现 O(1) 的查找
- 边遍历边存，避免两次遍历
- 注意返回的是索引，不是数值
