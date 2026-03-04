# 滑动窗口（Sliding Window）

---

## 1. 无重复字符的最长子串（Longest Substring Without Repeating Characters）

**题号**：003  
**思路**：双指针 + 哈希集合，右扩左缩  
**代码**：
```python
def lengthOfLongestSubstring(s):
    char_set = set()
    left = max_len = 0
    for right in range(len(s)):
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1
        char_set.add(s[right])
        max_len = max(max_len, right - left + 1)
    return max_len
```
**复杂度**：时间 O(n)，空间 O(min(m, n))

---

## 2. 找到字符串中所有字母异位词（Find All Anagrams in a String）

**题号**：438  
**思路**：固定窗口大小，滑动比较字符计数  
**代码**：
```python
def findAnagrams(s, p):
    from collections import Counter
    p_count = Counter(p)
    s_count = Counter()
    res = []
    for i, c in enumerate(s):
        s_count[c] += 1
        if i >= len(p):
            if s_count[s[i - len(p)]] == 1:
                del s_count[s[i - len(p)]]
            else:
                s_count[s[i - len(p)]] -= 1
        if s_count == p_count:
            res.append(i - len(p) + 1)
    return res
```
**复杂度**：时间 O(n)，空间 O(1)（字符集固定26个）

---

## 3. 和为 K 的子数组（Subarray Sum Equals K）

**题号**：560  
**思路**：前缀和 + 哈希表，`prefix[j] - prefix[i] = k`  
**代码**：
```python
def subarraySum(nums, k):
    from collections import defaultdict
    prefix_sum = defaultdict(int)
    prefix_sum[0] = 1
    curr_sum = count = 0
    for num in nums:
        curr_sum += num
        count += prefix_sum[curr_sum - k]
        prefix_sum[curr_sum] += 1
    return count
```
**复杂度**：时间 O(n)，空间 O(n)

---

## 4. 滑动窗口最大值（Sliding Window Maximum）

**题号**：239  
**思路**：单调队列，队首维护最大值  
**代码**：
```python
def maxSlidingWindow(nums, k):
    from collections import deque
    q = deque()  # 存储索引
    res = []
    for i, num in enumerate(nums):
        # 移除队尾较小元素
        while q and nums[q[-1]] < num:
            q.pop()
        q.append(i)
        # 移除窗口外元素
        if q[0] <= i - k:
            q.popleft()
        # 窗口形成后开始记录
        if i >= k - 1:
            res.append(nums[q[0]])
    return res
```
**复杂度**：时间 O(n)，空间 O(k)

---

## 5. 最小覆盖子串（Minimum Window Substring）

**题号**：076  
**思路**：双指针 + 字符计数，收缩左边界找最小窗口  
**代码**：
```python
def minWindow(s, t):
    from collections import Counter
    need = Counter(t)
    missing = len(t)
    left = start = 0
    min_len = float('inf')
    
    for right, c in enumerate(s, 1):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        
        while missing == 0:
            if right - left < min_len:
                min_len = right - left
                start = left
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    
    return s[start:start + min_len] if min_len != float('inf') else ""
```
**复杂度**：时间 O(|s| + |t|)，空间 O(|t|)
