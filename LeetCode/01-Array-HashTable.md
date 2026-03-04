# 数组与哈希表（Array & HashTable）

---

## 1. 两数之和（Two Sum）

**题号**：001  
**难度**：简单

### 题目描述
给定一个整数数组 `nums` 和一个整数目标值 `target`，请你在该数组中找出 **和为目标值** 的那 **两个** 整数，并返回它们的数组下标。

你可以假设每种输入只会对应一个答案。但是，数组中同一个元素在答案里不能重复出现。

### 示例
```
输入：nums = [2,7,11,15], target = 9
输出：[0,1]
解释：因为 nums[0] + nums[1] == 9，返回 [0, 1]
```

### 解题思路
使用哈希表存储已遍历的数字及其索引。遍历数组时，检查 `target - nums[i]` 是否在哈希表中。

### 代码实现
```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```

### 复杂度分析
- **时间复杂度**：O(n) - 只需遍历数组一次
- **空间复杂度**：O(n) - 哈希表存储 n 个元素

---

## 2. 字母异位词分组（Group Anagrams）

**题号**：049  
**难度**：中等

### 题目描述
给你一个字符串数组，请你将 **字母异位词** 组合在一起。可以按任意顺序返回结果列表。

**字母异位词** 是由重新排列源单词的字母得到的一个新单词，所有源单词中的字母通常恰好只用一次。

### 示例
```
输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
```

### 解题思路
排序后的字符串作为 key，哈希表分组。

### 代码实现
```python
def groupAnagrams(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

### 复杂度分析
- **时间复杂度**：O(n·k·logk) - n 是字符串数量，k 是字符串最大长度
- **空间复杂度**：O(n·k)

---

## 3. 最长连续序列（Longest Consecutive Sequence）

**题号**：128  
**难度**：中等

### 题目描述
给定一个未排序的整数数组 `nums`，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。

请你设计并实现时间复杂度为 O(n) 的算法解决此问题。

### 示例
```
输入：nums = [100,4,200,1,3,2]
输出：4
解释：最长数字连续序列是 [1, 2, 3, 4]，长度为 4
```

### 解题思路
哈希集合，只从序列起点开始计数（即 num-1 不在集合中的数）。

### 代码实现
```python
def longestConsecutive(nums):
    num_set = set(nums)
    longest = 0
    for num in num_set:
        if num - 1 not in num_set:  # 只从起点开始
            curr, streak = num, 1
            while curr + 1 in num_set:
                curr += 1
                streak += 1
            longest = max(longest, streak)
    return longest
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 4. 移动零（Move Zeroes）

**题号**：283  
**难度**：简单

### 题目描述
给定一个数组 `nums`，编写一个函数将所有 `0` 移动到数组的末尾，同时保持非零元素的相对顺序。

**请注意**，必须在不复制数组的情况下原地对数组进行操作。

### 示例
```
输入: nums = [0,1,0,3,12]
输出: [1,3,12,0,0]
```

### 解题思路
双指针，非零元素前移，后面补零。

### 代码实现
```python
def moveZeroes(nums):
    j = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[j], nums[i] = nums[i], nums[j]
            j += 1
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 5. 盛最多水的容器（Container With Most Water）

**题号**：011  
**难度**：中等

### 题目描述
给定一个长度为 `n` 的整数数组 `height`。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])`。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。返回容器可以储存的最大水量。

**说明**：你不能倾斜容器。

### 示例
```
输入：[1,8,6,2,5,4,8,3,7]
输出：49
解释：选择第 1 条线（height=8）和第 8 条线（height=7），面积为 min(8,7) * (8-1) = 49
```

### 解题思路
双指针，移动较短的一边，因为面积受限于较短边。

### 代码实现
```python
def maxArea(height):
    left, right = 0, len(height) - 1
    max_area = 0
    while left < right:
        width = right - left
        if height[left] < height[right]:
            max_area = max(max_area, height[left] * width)
            left += 1
        else:
            max_area = max(max_area, height[right] * width)
            right -= 1
    return max_area
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 6. 三数之和（3Sum）

**题号**：015  
**难度**：中等

### 题目描述
给你一个整数数组 `nums`，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k`，同时还满足 `nums[i] + nums[j] + nums[k] == 0`。

请你返回所有和为 `0` 且不重复的三元组。

**注意**：答案中不可以包含重复的三元组。

### 示例
```
输入：nums = [-1,0,1,2,-1,-4]
输出：[[-1,-1,2],[-1,0,1]]
```

### 解题思路
排序 + 双指针，去重是关键。

### 代码实现
```python
def threeSum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i-1]:
            continue
        left, right = i + 1, len(nums) - 1
        while left < right:
            total = nums[i] + nums[left] + nums[right]
            if total < 0: left += 1
            elif total > 0: right -= 1
            else:
                res.append([nums[i], nums[left], nums[right]])
                while left < right and nums[left] == nums[left+1]: left += 1
                while left < right and nums[right] == nums[right-1]: right -= 1
                left += 1; right -= 1
    return res
```

### 复杂度分析
- **时间复杂度**：O(n²)
- **空间复杂度**：O(1)（不计结果）

---

## 7. 接雨水（Trapping Rain Water）

**题号**：042  
**难度**：困难

### 题目描述
给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

### 示例
```
输入：height = [0,1,0,2,1,0,1,3,2,1,2,1]
输出：6
解释：可以接 6 个单位的雨水
```

### 解题思路
双指针，维护左右最大高度，计算当前位置能接的雨水量。

### 代码实现
```python
def trap(height):
    left, right = 0, len(height) - 1
    left_max = right_max = water = 0
    while left < right:
        if height[left] < height[right]:
            left_max = max(left_max, height[left])
            water += left_max - height[left]
            left += 1
        else:
            right_max = max(right_max, height[right])
            water += right_max - height[right]
            right -= 1
    return water
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)
