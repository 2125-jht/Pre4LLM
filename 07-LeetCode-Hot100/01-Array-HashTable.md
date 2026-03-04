# 数组与哈希表（Array & HashTable）

---

## 1. 两数之和（Two Sum）

**题号**：001  
**思路**：哈希表存储已遍历数字，查找 `target - nums[i]`  
**代码**：
```python
def twoSum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        if target - num in seen:
            return [seen[target - num], i]
        seen[num] = i
```
**复杂度**：时间 O(n)，空间 O(n)

---

## 2. 字母异位词分组（Group Anagrams）

**题号**：049  
**思路**：排序后的字符串作为 key，哈希表分组  
**代码**：
```python
def groupAnagrams(strs):
    from collections import defaultdict
    groups = defaultdict(list)
    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```
**复杂度**：时间 O(n·k·logk)，空间 O(n·k)

---

## 3. 最长连续序列（Longest Consecutive Sequence）

**题号**：128  
**思路**：哈希集合，只从序列起点开始计数  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(n)

---

## 4. 移动零（Move Zeroes）

**题号**：283  
**思路**：双指针，非零元素前移，后面补零  
**代码**：
```python
def moveZeroes(nums):
    j = 0
    for i in range(len(nums)):
        if nums[i] != 0:
            nums[j], nums[i] = nums[i], nums[j]
            j += 1
```
**复杂度**：时间 O(n)，空间 O(1)

---

## 5. 盛最多水的容器（Container With Most Water）

**题号**：011  
**思路**：双指针，移动较短的一边  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(1)

---

## 6. 三数之和（3Sum）

**题号**：015  
**思路**：排序 + 双指针，去重是关键  
**代码**：
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
**复杂度**：时间 O(n²)，空间 O(1)（不计结果）

---

## 7. 接雨水（Trapping Rain Water）

**题号**：042  
**思路**：双指针，维护左右最大高度  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(1)
