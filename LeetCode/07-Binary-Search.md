# 二分查找（Binary Search）

---

## 1. 搜索插入位置（Search Insert Position）

**题号**：035  
**难度**：简单

### 题目描述
给定一个排序数组和一个目标值，在数组中找到目标值，并返回其索引。如果目标值不存在于数组中，返回它将会被按顺序插入的位置。

请必须使用时间复杂度为 `O(log n)` 的算法。

### 示例
```
输入：nums = [1,3,5,6], target = 5
输出：2

输入：nums = [1,3,5,6], target = 2
输出：1
解释：2 不在数组中，应该插入在索引 1 的位置

输入：nums = [1,3,5,6], target = 7
输出：4
```

### 解题思路
标准二分查找，返回 left 即为插入位置。

### 代码实现
```python
def searchInsert(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    return left
```

### 复杂度分析
- **时间复杂度**：O(log n)
- **空间复杂度**：O(1)

---

## 2. 在排序数组中查找元素的第一个和最后一个位置（Find First and Last Position of Element in Sorted Array）

**题号**：034  
**难度**：中等

### 题目描述
给你一个按照非降序排列的整数数组 `nums`，和一个目标值 `target`。请你找出给定目标值在数组中的开始位置和结束位置。

如果数组中不存在目标值 `target`，返回 `[-1, -1]`。

你必须设计并实现时间复杂度为 `O(log n)` 的算法解决此问题。

### 示例
```
输入：nums = [5,7,7,8,8,10], target = 8
输出：[3,4]

输入：nums = [5,7,7,8,8,10], target = 6
输出：[-1,-1]

输入：nums = [], target = 0
输出：[-1,-1]
```

### 解题思路
使用两次二分查找：
1. 第一次查找目标值的**左边界**（第一个出现的位置）
2. 第二次查找目标值的**右边界**（最后一个出现的位置）

查找左边界时，当 `nums[mid] == target`，继续向左搜索（`right = mid - 1`）。  
查找右边界时，当 `nums[mid] == target`，继续向右搜索（`left = mid + 1`）。

### 代码实现
```python
def searchRange(nums, target):
    def findLeft():
        left, right = 0, len(nums) - 1
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                ans = mid
                right = mid - 1  # 继续向左找
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return ans
    
    def findRight():
        left, right = 0, len(nums) - 1
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                ans = mid
                left = mid + 1   # 继续向右找
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return ans
    
    return [findLeft(), findRight()]
```

### 复杂度分析
- **时间复杂度**：O(log n)，进行了两次二分查找
- **空间复杂度**：O(1)

---

## 3. 搜索旋转排序数组（Search in Rotated Sorted Array）

**题号**：033  
**难度**：中等

### 题目描述
整数数组 `nums` 按升序排列，数组中的值 **互不相同**。

在传递给函数之前，`nums` 在预先未知的某个下标 `k`（`0 <= k < nums.length`）上进行了 **旋转**，使数组变为 `[nums[k], nums[k+1], ..., nums[n-1], nums[0], nums[1], ..., nums[k-1]]`（下标 **从 0 开始** 计数）。

给你 **旋转后** 的数组 `nums` 和一个整数 `target`，如果 `nums` 中存在这个目标值 `target`，则返回它的下标，否则返回 `-1`。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

### 示例
```
输入：nums = [4,5,6,7,0,1,2], target = 0
输出：4

输入：nums = [4,5,6,7,0,1,2], target = 3
输出：-1

输入：nums = [1], target = 0
输出：-1
```

### 解题思路
判断哪一半是有序的，再判断 target 是否在有序区间内。

### 代码实现
```python
def search(nums, target):
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        
        # 左半部分有序
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        # 右半部分有序
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1
```

### 复杂度分析
- **时间复杂度**：O(log n)
- **空间复杂度**：O(1)

---

## 4. 寻找旋转排序数组中的最小值（Find Minimum in Rotated Sorted Array）

**题号**：153  
**难度**：中等

### 题目描述
已知一个长度为 `n` 的数组，预先按照升序排列，经由 `1` 到 `n` 次 **旋转** 后，得到输入数组。

例如，原数组 `nums = [0,1,2,4,5,6,7]` 在变化后可能得到：
- 若旋转 `4` 次，则可以得到 `[4,5,6,7,0,1,2]`
- 若旋转 `7` 次，则可以得到 `[0,1,2,4,5,6,7]`

给你一个元素值 **互不相同** 的数组 `nums`，它原来是一个升序排列的数组，并按上述情形进行了多次旋转。请你找出并返回数组中的 **最小元素**。

你必须设计一个时间复杂度为 `O(log n)` 的算法解决此问题。

### 示例
```
输入：nums = [3,4,5,1,2]
输出：1
解释：原数组为 [1,2,3,4,5] ，旋转 3 次得到输入数组

输入：nums = [4,5,6,7,0,1,2]
输出：0
解释：原数组为 [0,1,2,4,5,6,7] ，旋转 4 次得到输入数组
```

### 解题思路
与右端点比较，判断最小值在哪一半。

### 代码实现
```python
def findMin(nums):
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]
```

### 复杂度分析
- **时间复杂度**：O(log n)
- **空间复杂度**：O(1)

---

## 5. 寻找两个正序数组的中位数（Median of Two Sorted Arrays）

**题号**：004  
**难度**：困难

### 题目描述
给定两个大小分别为 `m` 和 `n` 的正序（从小到大）数组 `nums1` 和 `nums2`。请你找出并返回这两个正序数组的 **中位数**。

算法的时间复杂度应该为 `O(log (m+n))`。

### 示例
```
输入：nums1 = [1,3], nums2 = [2]
输出：2.00000
解释：合并数组 = [1,2,3]，中位数 2

输入：nums1 = [1,2], nums2 = [3,4]
输出：2.50000
解释：合并数组 = [1,2,3,4]，中位数 (2 + 3) / 2 = 2.5
```

### 解题思路
使用「寻找第 K 小元素」的思路：

1. **核心思想**：每次比较两个数组中第 `k/2` 个元素，排除掉较小的那一半（它们不可能是第 k 小的元素）
2. **具体做法**：
   - 在 `nums1` 和 `nums2` 中分别取前 `k/2` 个元素
   - 如果 `nums1[k/2-1] < nums2[k/2-1]`，说明 `nums1` 的前半部分都小于第 k 个元素，可以全部排除
   - 更新 `k` 值和数组起始位置，继续查找
3. **终止条件**：
   - 当 `k == 1` 时，返回两个数组当前起始位置的最小值
   - 当某个数组被耗尽时，直接在另一个数组中取第 k 个元素

### 代码实现
```python
def findMedianSortedArrays(nums1, nums2):
    def findKth(k):
        start1, start2 = 0, 0
        while True:
            if start1 == m:
                return nums2[start2 + k - 1]
            if start2 == n:
                return nums1[start1 + k - 1]
            if k == 1:
                return min(nums1[start1], nums2[start2])
            
            length = k // 2
            i1 = min(start1 + length - 1, m - 1)
            i2 = min(start2 + length - 1, n - 1)
            if nums1[i1] < nums2[i2]:
                k = k - (i1 - start1 + 1)
                start1 = i1 + 1
            else:
                k = k - (i2 - start2 + 1)
                start2 = i2 + 1
    
    m, n = len(nums1), len(nums2)
    total = m + n
    if total % 2 == 0:
        return (findKth(total // 2) + findKth(total // 2 + 1)) / 2
    else:
        return findKth(total // 2 + 1)
```

### 复杂度分析
- **时间复杂度**：O(log(min(m,n)))
- **空间复杂度**：O(1)
