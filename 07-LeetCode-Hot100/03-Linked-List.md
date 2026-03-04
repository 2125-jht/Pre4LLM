# 链表（Linked List）

---

## 1. 两数相加（Add Two Numbers）

**题号**：002  
**思路**：模拟加法，注意进位和链表长度不等  
**代码**：
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
**复杂度**：时间 O(max(m,n))，空间 O(max(m,n))

---

## 2. 合并两个有序链表（Merge Two Sorted Lists）

**题号**：021  
**思路**：递归或迭代，每次选较小节点  
**代码**（迭代）：
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
**复杂度**：时间 O(m+n)，空间 O(1)

---

## 3. 反转链表（Reverse Linked List）

**题号**：206  
**思路**：三指针迭代，逐个反转指向  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(1)

---

## 4. 删除链表的倒数第 N 个结点（Remove Nth Node From End）

**题号**：019  
**思路**：快慢指针，先让快指针走 N 步  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(1)

---

## 5. 合并 K 个升序链表（Merge k Sorted Lists）

**题号**：023  
**思路**：优先队列（堆）或分治合并  
**代码**（优先队列）：
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
**复杂度**：时间 O(N·logk)，空间 O(k)

---

## 6. 环形链表（Linked List Cycle）

**题号**：141  
**思路**：快慢指针，相遇则有环  
**代码**：
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
**复杂度**：时间 O(n)，空间 O(1)

---

## 7. 相交链表（Intersection of Two Linked Lists）

**题号**：160  
**思路**：双指针交替遍历，消除长度差  
**代码**：
```python
def getIntersectionNode(headA, headB):
    pA, pB = headA, headB
    while pA != pB:
        pA = pA.next if pA else headB
        pB = pB.next if pB else headA
    return pA
```
**复杂度**：时间 O(m+n)，空间 O(1)
