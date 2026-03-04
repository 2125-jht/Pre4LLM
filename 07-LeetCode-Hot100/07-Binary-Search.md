# 二分查找（Binary Search）

---

## 1. 搜索插入位置（Search Insert Position）

**题号**：035  
**思路**：标准二分查找，返回 left 即为插入位置  
**代码**：
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
**复杂度**：时间 O(log n)，空间 O(1)

---

## 2. 搜索旋转排序数组（Search in Rotated Sorted Array）

**题号**：033  
**思路**：判断哪一半是有序的，再判断 target 是否在有序区间内  
**代码**：
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
**复杂度**：时间 O(log n)，空间 O(1)

---

## 3. 寻找旋转排序数组中的最小值（Find Minimum in Rotated Sorted Array）

**题号**：153  
**思路**：与右端点比较，判断最小值在哪一半  
**代码**：
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
**复杂度**：时间 O(log n)，空间 O(1)

---

## 4. 寻找两个正序数组的中位数（Median of Two Sorted Arrays）

**题号**：004  
**思路**：二分查找分割线，使得左右两部分元素个数相等且左边最大值 <= 右边最小值  
**代码**：
```python
def findMedianSortedArrays(nums1, nums2):
    if len(nums1) > len(nums2):
        nums1, nums2 = nums2, nums1
    
    m, n = len(nums1), len(nums2)
    left, right = 0, m
    
    while left <= right:
        partition1 = (left + right) // 2
        partition2 = (m + n + 1) // 2 - partition1
        
        max_left1 = float('-inf') if partition1 == 0 else nums1[partition1 - 1]
        min_right1 = float('inf') if partition1 == m else nums1[partition1]
        max_left2 = float('-inf') if partition2 == 0 else nums2[partition2 - 1]
        min_right2 = float('inf') if partition2 == n else nums2[partition2]
        
        if max_left1 <= min_right2 and max_left2 <= min_right1:
            if (m + n) % 2 == 0:
                return (max(max_left1, max_left2) + min(min_right1, min_right2)) / 2
            else:
                return max(max_left1, max_left2)
        elif max_left1 > min_right2:
            right = partition1 - 1
        else:
            left = partition1 + 1
```
**复杂度**：时间 O(log(min(m,n)))，空间 O(1)
