# 链表（Linked List）

---

## 1. 两数相加（Add Two Numbers）

**题号**：002  
**难度**：中等

### 题目描述
给你两个 **非空** 的链表，表示两个非负的整数。它们每位数字都是按照 **逆序** 的方式存储的，并且每个节点只能存储 **一位** 数字。

请你将两个数相加，并以相同形式返回一个表示和的链表。

你可以假设除了数字 0 之外，这两个数都不会以 0 开头。

### 示例
```
输入：l1 = [2,4,3], l2 = [5,6,4]
输出：[7,0,8]
解释：342 + 465 = 807

输入：l1 = [9,9,9,9,9,9,9], l2 = [9,9,9,9]
输出：[8,9,9,9,0,0,0,1]
```

### 解题思路
模拟加法，注意进位和链表长度不等。

### 代码实现
```python
def addTwoNumbers(l1, l2):
    dummy = ListNode(0)
    curr, carry = dummy, 0
    while l1 or l2 or carry:
        val1 = l1.val if l1 else 0
        val2 = l2.val if l2 else 0
        total = val1 + val2 + carry
        carry, val = divmod(total, 10)
        curr.next = ListNode(val)
        curr = curr.next
        if l1: l1 = l1.next
        if l2: l2 = l2.next
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(max(m,n))
- **空间复杂度**：O(max(m,n))

---

## 2. 合并两个有序链表（Merge Two Sorted Lists）

**题号**：021  
**难度**：简单

### 题目描述
将两个升序链表合并为一个新的 **升序** 链表并返回。新链表是通过拼接给定的两个链表的所有节点组成的。

### 示例
```
输入：l1 = [1,2,4], l2 = [1,3,4]
输出：[1,1,2,3,4,4]
```

### 解题思路
递归或迭代，每次选较小节点。

### 代码实现
```python
def mergeTwoLists(l1, l2):
    dummy = ListNode(0)
    curr = dummy
    while l1 and l2:
        if l1.val < l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next
    curr.next = l1 or l2
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(m+n)
- **空间复杂度**：O(1)

---

## 3. 反转链表（Reverse Linked List）

**题号**：206  
**难度**：简单

### 题目描述
给你单链表的头节点 `head`，请你反转链表，并返回反转后的链表。

### 示例
```
输入：head = [1,2,3,4,5]
输出：[5,4,3,2,1]

输入：head = [1,2]
输出：[2,1]
```

### 解题思路
三指针迭代，逐个反转指向。

### 代码实现
```python
def reverseList(head):
    prev, curr = None, head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 4. 删除链表的倒数第 N 个结点（Remove Nth Node From End）

**题号**：019  
**难度**：中等

### 题目描述
给你一个链表，删除链表的倒数第 `n` 个结点，并且返回链表的头结点。

### 示例
```
输入：head = [1,2,3,4,5], n = 2
输出：[1,2,3,5]

输入：head = [1], n = 1
输出：[]
```

### 解题思路
快慢指针，先让快指针走 N 步。

### 代码实现
```python
def removeNthFromEnd(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n + 1):
        fast = fast.next
    while fast:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 5. 合并 K 个升序链表（Merge k Sorted Lists）

**题号**：023  
**难度**：困难

### 题目描述
给你一个链表数组，每个链表都已经按升序排列。

请你将所有链表合并到一个升序链表中，返回合并后的链表。

### 示例
```
输入：lists = [[1,4,5],[1,3,4],[2,6]]
输出：[1,1,2,3,4,4,5,6]
解释：链表数组如下：
[
  1->4->5,
  1->3->4,
  2->6
]
将它们合并到一个有序链表中得到：
1->1->2->3->4->4->5->6
```

### 解题思路
优先队列（堆）或分治合并。

### 代码实现
```python
def mergeKLists(lists):
    import heapq
    dummy = ListNode(0)
    curr = dummy
    heap = []
    for i, l in enumerate(lists):
        if l:
            heapq.heappush(heap, (l.val, i, l))
    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(N·logk)，N 是所有节点总数
- **空间复杂度**：O(k)

---

## 6. 环形链表（Linked List Cycle）

**题号**：141  
**难度**：简单

### 题目描述
给你一个链表的头节点 `head`，判断链表中是否有环。

如果链表中有某个节点，可以通过连续跟踪 `next` 指针再次到达，则链表中存在环。

### 示例
```
输入：head = [3,2,0,-4], pos = 1
输出：true
解释：链表中有一个环，其尾部连接到第二个节点
```

### 解题思路
快慢指针，相遇则有环。

### 代码实现
```python
def hasCycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 7. 相交链表（Intersection of Two Linked Lists）

**题号**：160  
**难度**：简单

### 题目描述
给你两个单链表的头节点 `headA` 和 `headB`，请你找出并返回两个单链表相交的起始节点。如果两个链表不存在相交节点，返回 `null`。

### 示例
```
输入：intersectVal = 8, listA = [4,1,8,4,5], listB = [5,6,1,8,4,5]
输出：Intersected at '8'
解释：相交节点的值为 8（注意，如果两个链表相交则不能为 0）
```

### 解题思路
双指针交替遍历，消除长度差，数学证明两指针走的总路程相同。

### 代码实现
```python
def getIntersectionNode(headA, headB):
    pA, pB = headA, headB
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    return pA
```

### 复杂度分析
- **时间复杂度**：O(m+n)
- **空间复杂度**：O(1)

---

## 8. 排序链表（Sort List）

**题号**：148  
**难度**：中等

### 题目描述
给你链表的头结点 `head`，请将其按 **升序** 排列并返回 **排序后的链表**。

### 示例
```
输入：head = [4,2,1,3]
输出：[1,2,3,4]

输入：head = [-1,5,3,4,0]
输出：[-1,0,3,4,5]
```

### 解题思路
归并排序，快慢指针找中点，递归排序左右两部分后合并。

### 代码实现
```python
def sortList(head):
    if not head or not head.next:
        return head
    
    # 快慢指针找中点
    slow, fast = head, head.next
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    mid = slow.next
    slow.next = None
    
    # 递归排序
    left = sortList(head)
    right = sortList(mid)
    
    # 合并两个有序链表
    dummy = ListNode(0)
    curr = dummy
    while left and right:
        if left.val < right.val:
            curr.next = left
            left = left.next
        else:
            curr.next = right
            right = right.next
        curr = curr.next
    curr.next = left or right
    
    return dummy.next
```

### 复杂度分析
- **时间复杂度**：O(n·logn)
- **空间复杂度**：O(logn) - 递归栈空间

---

## 9. 随机链表的复制（Copy List with Random Pointer）

**题号**：138  
**难度**：中等

### 题目描述
给你一个长度为 `n` 的链表，每个节点包含一个额外增加的随机指针 `random`，该指针可以指向链表中的任何节点或空节点。

构造这个链表的 **深拷贝**。深拷贝应该正好由 `n` 个 **全新** 节点组成，其中每个新节点的值与其对应的原节点的值相同。

### 示例
```
输入：head = [[7,null],[13,0],[11,4],[10,2],[1,0]]
输出：[[7,null],[13,0],[11,4],[10,2],[1,0]]
```

### 解题思路
哈希表存储原节点到新节点的映射，两次遍历完成复制。

### 代码实现
```python
def copyRandomList(head):
    if not head:
        return None
    
    # 第一次遍历：创建新节点并建立映射
    old_to_new = {}
    curr = head
    while curr:
        old_to_new[curr] = ListNode(curr.val)
        curr = curr.next
    
    # 第二次遍历：设置 next 和 random 指针
    curr = head
    while curr:
        if curr.next:
            old_to_new[curr].next = old_to_new[curr.next]
        if curr.random:
            old_to_new[curr].random = old_to_new[curr.random]
        curr = curr.next
    
    return old_to_new[head]
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(n)

---

## 10. 回文链表（Palindrome Linked List）

**题号**：234  
**难度**：简单

### 题目描述
给你一个单链表的头节点 `head`，请判断该链表是否为回文链表。如果是，返回 `true`；否则，返回 `false`。

### 示例
```
输入：head = [1,2,2,1]
输出：true

输入：head = [1,2]
输出：false
```

### 解题思路
快慢指针找中点，反转后半部分，然后比较前后两部分。

### 代码实现
```python
def isPalindrome(head):
    if not head or not head.next:
        return True
    
    # 快慢指针找中点
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    
    # 反转后半部分
    prev = None
    while slow:
        next_temp = slow.next
        slow.next = prev
        prev = slow
        slow = next_temp
    
    # 比较前后两部分
    left, right = head, prev
    while right:  # 后半部分可能更短
        if left.val != right.val:
            return False
        left = left.next
        right = right.next
    
    return True
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)

---

## 11. 奇偶链表（Odd Even Linked List）

**题号**：328  
**难度**：中等

### 题目描述
给定单链表的头节点 `head`，将所有索引为奇数的节点和索引为偶数的节点分别组合在一起，然后返回重新排序的列表。

**第一个**节点的索引被认为是 **奇数**，**第二个**节点的索引为 **偶数**，以此类推。

请注意，偶数组和奇数组内部的相对顺序应该与输入时保持一致。

### 示例
```
输入: head = [1,2,3,4,5]
输出: [1,3,5,2,4]

输入: head = [2,1,3,5,6,4,7]
输出: [2,3,6,7,1,5,4]
```

### 解题思路
分离奇偶节点，然后将偶链表接到奇链表后面。

### 代码实现
```python
def oddEvenList(head):
    if not head or not head.next:
        return head
    
    odd = head           # 奇数索引节点头
    even = head.next     # 偶数索引节点头
    even_head = even     # 保存偶数头
    
    while even and even.next:
        odd.next = even.next
        odd = odd.next
        even.next = odd.next
        even = even.next
    
    odd.next = even_head
    return head
```

### 复杂度分析
- **时间复杂度**：O(n)
- **空间复杂度**：O(1)
