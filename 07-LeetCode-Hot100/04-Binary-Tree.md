# 二叉树（Binary Tree）

---

## 1. 二叉树的中序遍历（Binary Tree Inorder Traversal）

**题号**：094  
**思路**：递归或栈迭代，左-根-右  
**代码**（迭代）：
```python
def inorderTraversal(root):
    res, stack = [], []
    curr = root
    while curr or stack:
        while curr:
            stack.append(curr)
            curr = curr.left
        curr = stack.pop()
        res.append(curr.val)
        curr = curr.right
    return res
```
**复杂度**：时间 O(n)，空间 O(n)

---

## 2. 二叉树的最大深度（Maximum Depth of Binary Tree）

**题号**：104  
**思路**：递归，深度 = max(左子树深度, 右子树深度) + 1  
**代码**：
```python
def maxDepth(root):
    if not root:
        return 0
    return 1 + max(maxDepth(root.left), maxDepth(root.right))
```
**复杂度**：时间 O(n)，空间 O(h)（h为树高）

---

## 3. 翻转二叉树（Invert Binary Tree）

**题号**：226  
**思路**：递归交换左右子树  
**代码**：
```python
def invertTree(root):
    if not root:
        return None
    root.left, root.right = invertTree(root.right), invertTree(root.left)
    return root
```
**复杂度**：时间 O(n)，空间 O(h)

---

## 4. 二叉树的直径（Diameter of Binary Tree）

**题号**：543  
**思路**：后序遍历，直径 = 左深度 + 右深度  
**代码**：
```python
def diameterOfBinaryTree(root):
    self.diameter = 0
    def depth(node):
        if not node:
            return 0
        left = depth(node.left)
        right = depth(node.right)
        self.diameter = max(self.diameter, left + right)
        return 1 + max(left, right)
    depth(root)
    return self.diameter
```
**复杂度**：时间 O(n)，空间 O(h)

---

## 5. 二叉树的层序遍历（Binary Tree Level Order Traversal）

**题号**：102  
**思路**：BFS，队列逐层处理  
**代码**：
```python
def levelOrder(root):
    if not root:
        return []
    from collections import deque
    queue, res = deque([root]), []
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left: queue.append(node.left)
            if node.right: queue.append(node.right)
        res.append(level)
    return res
```
**复杂度**：时间 O(n)，空间 O(n)

---

## 6. 将有序数组转换为二叉搜索树（Convert Sorted Array to BST）

**题号**：108  
**思路**：二分递归，中间元素作为根  
**代码**：
```python
def sortedArrayToBST(nums):
    if not nums:
        return None
    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sortedArrayToBST(nums[:mid])
    root.right = sortedArrayToBST(nums[mid+1:])
    return root
```
**复杂度**：时间 O(n)，空间 O(log n)

---

## 7. 验证二叉搜索树（Validate Binary Search Tree）

**题号**：098  
**思路**：中序遍历应为递增序列，或递归判断范围  
**代码**（递归范围）：
```python
def isValidBST(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (isValidBST(root.left, min_val, root.val) and
            isValidBST(root.right, root.val, max_val))
```
**复杂度**：时间 O(n)，空间 O(h)

---

## 8. 二叉树展开为链表（Flatten Binary Tree to Linked List）

**题号**：114  
**思路**：递归，将左右子树分别展开后拼接  
**代码**：
```python
def flatten(root):
    if not root:
        return
    flatten(root.left)
    flatten(root.right)
    
    temp = root.right
    root.right = root.left
    root.left = None
    
    curr = root
    while curr.right:
        curr = curr.right
    curr.right = temp
```
**复杂度**：时间 O(n)，空间 O(h)
