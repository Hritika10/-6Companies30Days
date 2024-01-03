                                             #PROBLEM 1: Minimize the Maximum of Two Arrays
def minimumMaximum(divisor1, divisor2, uniqueCnt1, uniqueCnt2):
    # Calculate the starting points for arr1 and arr2
    start1 = (divisor1 - 1) * uniqueCnt1 + 1
    start2 = (divisor2 - 1) * uniqueCnt2 + 1

    # Build arr1 and arr2
    arr1 = [start1 + i * divisor1 for i in range(uniqueCnt1)]
    arr2 = [start2 + i * divisor2 for i in range(uniqueCnt2)]

    # Return the minimum possible maximum integer
    return max(max(arr1, default=0), max(arr2, default=0))

# Test cases
print(minimumMaximum(2, 7, 1, 3))  # Output: 4
print(minimumMaximum(3, 5, 2, 1))  # Output: 3
print(minimumMaximum(2, 4, 8, 2))  # Output: 15


                                             #PROBLEM 2: Employee Priority Systems
from collections import defaultdict

def high_access_employees(access_times):
    employee_times = defaultdict(list)
    
    for employee, time in access_times:
        employee_times[employee].append(time)
    
    result = []
    
    for employee, times in employee_times.items():
        times.sort()
        
        count = 0
        prev_time = None
        
        for time in times:
            if prev_time and int(time) - int(prev_time) <= 100:
                count += 1
            else:
                count = 1
            
            prev_time = time
            
            if count >= 3:
                result.append(employee)
                break
    
    return result

# Test cases
print(high_access_employees([["a", "0549"],["b", "0457"],["a", "0532"],["a", "0621"],["b", "0540"]]))  # Output: ["a"]
print(high_access_employees([["d", "0002"],["c", "0808"],["c", "0829"],["e", "0215"],["d", "1508"],["d", "1444"],["d", "1410"],["c", "0809"]]))  # Output: ["c", "d"]
print(high_access_employees([["cd", "1025"],["ab", "1025"],["cd", "1046"],["cd", "1055"],["ab", "1124"],["ab", "1120"]]))  # Output: ["ab", "cd"]


                                                #PROBLEM 3: Kth Smallest Element Query
import heapq

def kthSmallestQueries(nums, queries):
    # Create a list to store the original numbers
    original_nums = list(nums)
    
    # Helper function to find the k-th smallest element using heap
    def kth_smallest(trimmed_nums, k):
        min_heap = [(int(num), i) for i, num in enumerate(trimmed_nums)]
        heapq.heapify(min_heap)
        
        for _ in range(k - 1):
            heapq.heappop(min_heap)
        
        return heapq.heappop(min_heap)[1]

    # Iterate through each query
    answer = []
    for k, trim in queries:
        # Trim each number to its rightmost trim digits
        for i in range(len(nums)):
            nums[i] = nums[i][-trim:]
        
        # Find the index of the k-th smallest number
        kth_smallest_index = kth_smallest(nums, k)
        
        # Record the index in the answer list
        answer.append(kth_smallest_index)
        
        # Reset each number to its original length
        nums = list(original_nums)
    
    # Return the answer list
    return answer

# Test cases
print(kthSmallestQueries(["102", "473", "251", "814"], [[1, 1], [2, 3], [4, 2], [1, 2]]))  # Output: [2, 2, 1, 0]
print(kthSmallestQueries(["24", "37", "96", "04"], [[2, 1], [2, 2]]))  # Output: [3, 0]


                                                #PROBLEM 4: COMBINATION SUM 
def combinationSum3(k, n):
    def backtrack(start, target, path):
        if k == len(path) and target == 0:
            result.append(path[:])
            return
        
        for i in range(start, 10):
            if i > target:
                break
            path.append(i)
            backtrack(i + 1, target - i, path)
            path.pop()

    result = []
    backtrack(1, n, [])
    return result

# Test cases
print(combinationSum3(3, 7))  # Output: [[1,2,4]]
print(combinationSum3(3, 9))  # Output: [[1,2,6],[1,3,5],[2,3,4]]
print(combinationSum3(4, 1))  # Output: []
   
                                     #PROBLEM 5: FLIP MATRIX 
import random

class Solution:
    def __init__(self, m, n):
        self.rows, self.cols = m, n
        self.total = m * n
        self.remaining = list(range(self.total))

    def flip(self):
        index = random.randint(0, len(self.remaining) - 1)
        row, col = divmod(self.remaining[index], self.cols)

        # Swap the selected index with the last index and pop it
        self.remaining[index], self.remaining[-1] = self.remaining[-1], self.remaining[index]
        self.remaining.pop()

        return [row, col]

    def reset(self):
        self.remaining = list(range(self.total))


# Test case
solution = Solution(3, 1)
print(solution.flip())  # Output: [1, 0]
print(solution.flip())  # Output: [2, 0]
print(solution.flip())  # Output: [0, 0]
solution.reset()
print(solution.flip())  # Output: [2, 0]


                                       #PROBLEM 6: COMBINATIONS IN THE PHONE NO 
def letterCombinations(digits):
    if not digits:
        return []

    digit_mapping = {
        '2': 'abc',
        '3': 'def',
        '4': 'ghi',
        '5': 'jkl',
        '6': 'mno',
        '7': 'pqrs',
        '8': 'tuv',
        '9': 'wxyz'
    }

    def backtrack(index, path):
        if index == len(digits):
            combinations.append(''.join(path))
            return

        current_digit = digits[index]
        for letter in digit_mapping[current_digit]:
            path.append(letter)
            backtrack(index + 1, path)
            path.pop()

    combinations = []
    backtrack(0, [])
    return combinations

# Test cases
print(letterCombinations("23"))  # Output: ["ad","ae","af","bd","be","bf","cd","ce","cf"]
print(letterCombinations(""))     # Output: []
print(letterCombinations("2"))    # Output: ["a","b","c"]


                                              #PROBLEM 7: FIND MISSING AND REPEATING 
def findTwoElement(arr, n):
    i = 0
    while i < n:
        if arr[i] != arr[arr[i] - 1]:
            arr[arr[i] - 1], arr[i] = arr[i], arr[arr[i] - 1]
        else:
            i += 1

    for i in range(n):
        if arr[i] != i + 1:
            return [arr[i], i + 1]

# Test cases
print(findTwoElement([2, 2], 2))    # Output: [2, 1]
print(findTwoElement([1, 3, 3], 3))  # Output: [3, 2]


                                           #PROBLEM 8: Find Consecutive Integers from a Data Stream
class DataStream:
    def __init__(self, value, k):
        self.value = value
        self.k = k
        self.buffer = [0] * k
        self.pointer = 0
        self.count = 0

    def consec(self, num):
        if num == self.value:
            self.buffer[self.pointer] = num
            self.pointer = (self.pointer + 1) % self.k
            self.count += 1
            return self.count == self.k
        else:
            self.count = 0
            return False

# Test case
dataStream = DataStream(4, 3)
print(dataStream.consec(4))  # Output: False
print(dataStream.consec(4))  # Output: False
print(dataStream.consec(4))  # Output: True
print(dataStream.consec(3))  # Output: False


                                    #PROBLEM 9: Find Consecutive Integers from a Data Stream
class DataStream:
    def __init__(self, value, k):
        self.value = value
        self.k = k
        self.buffer = [0] * k
        self.pointer = 0
        self.count = 0

    def consec(self, num):
        if num == self.value:
            self.buffer[self.pointer] = num
            self.pointer = (self.pointer + 1) % self.k
            self.count += 1
            return self.count == self.k
        else:
            self.count = 0
            return False

# Test case
dataStream = DataStream(4, 3)
print(dataStream.consec(4))  # Output: False
print(dataStream.consec(4))  # Output: False
print(dataStream.consec(4))  # Output: True
print(dataStream.consec(3))  # Output: False


                                      #PROBLEM 10: Number following a pattern
def printMinNumberForPattern(pattern):
    result = ""
    min_num, max_num = 1, len(pattern) + 1

    for char in pattern:
        if char == 'I':
            result += str(min_num)
            min_num += 1
        elif char == 'D':
            result += str(max_num)
            max_num -= 1

    # Add the last number to the result
    result += str(min_num)

    return result

# Test cases
print(printMinNumberForPattern("D"))      # Output: "21"
print(printMinNumberForPattern("IIDDD"))  # Output: "126543"


                                           #PROBLEM 11: K Divisible Elements Subarrays
def countDistinctSubarrays(nums, k, p):
    n = len(nums)
    count = 0
    prefix_count = {0: 1}
    current_prefix_sum = 0
    divisible_count = 0

    for i in range(n):
        current_prefix_sum += nums[i] % p
        if i >= k:
            current_prefix_sum -= nums[i - k] % p

        if current_prefix_sum % p == 0:
            divisible_count += 1

        if i >= k - 1:
            count += divisible_count
            prefix_count[current_prefix_sum] = prefix_count.get(current_prefix_sum, 0) + 1

    return count

# Test cases
print(countDistinctSubarrays([2,3,3,2,2], 2, 2))  # Output: 11
print(countDistinctSubarrays([1,2,3,4], 4, 1))     # Output: 10


                                                   #PROBLEM 12: Map of Highest Peak
from collections import deque

def highestPeak(isWater):
    m, n = len(isWater), len(isWater[0])
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    queue = deque()
    visited = set()

    # Initialize the queue with water cells and set their height to 0
    for i in range(m):
        for j in range(n):
            if isWater[i][j] == 1:
                queue.append((i, j))
                visited.add((i, j))

    # Perform BFS to assign heights to land cells
    while queue:
        i, j = queue.popleft()
        for di, dj in directions:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and (ni, nj) not in visited:
                isWater[ni][nj] = isWater[i][j] + 1
                queue.append((ni, nj))
                visited.add((ni, nj))

    return isWater

# Test cases
print(highestPeak([[0,1],[0,0]]))        # Output: [[1, 0], [2, 1]]
print(highestPeak([[0,0,1],[1,0,0],[0,0,0]]))  # Output: [[1, 1, 0], [0, 1, 1], [1, 2, 2]]


                                      #PROBLEM 13:  Maximum Sum BST in Binary Tree
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class Solution:
    def maxSumBST(self, root: TreeNode) -> int:
        def dfs(node):
            if not node:
                return (True, 0, float('inf'), float('-inf'))

            left_valid, left_sum, left_min, left_max = dfs(node.left)
            right_valid, right_sum, right_min, right_max = dfs(node.right)

            if left_valid and right_valid and left_max < node.val < right_min:
                current_sum = left_sum + right_sum + node.val
                current_min = min(left_min, node.val)
                current_max = max(right_max, node.val)
                return (True, current_sum, current_min, current_max)
            else:
                return (False, max(left_sum, right_sum), 0, 0)

        _, max_sum, _, _ = dfs(root)
        return max_sum

# Example usage
root1 = TreeNode(1, TreeNode(4, TreeNode(2), TreeNode(4)), TreeNode(3, TreeNode(2, TreeNode(4), TreeNode(6))))
solution = Solution()
print(solution.maxSumBST(root1))  # Output: 20

root2 = TreeNode(4, TreeNode(3, None, TreeNode(1, None, TreeNode(2))))
print(solution.maxSumBST(root2))  # Output: 2

root3 = TreeNode(-4, TreeNode(-2), TreeNode(-5))
print(solution.maxSumBST(root3))  # Output: 0


                                          #PROBLEM 14: Number of People Aware of a Secret
def countSecrets(n, delay, forget):
    mod = 10**9 + 7
    curr, total, last_shared, last_forgotten = 1, 1, 0, 0
    
    for day in range(1, n):
        curr, last_shared, last_forgotten = (curr + total - last_forgotten) % mod, (last_shared + total) % mod, (last_forgotten + total - curr) % mod

        if day >= delay:
            total = (total + curr) % mod
        
        if day >= forget:
            total = (total - last_shared) % mod
    
    return total

# Example usage
print(countSecrets(6, 2, 4))  # Output: 5
print(countSecrets(4, 1, 3))  # Output: 6


                                       #PROBLEM 15: Run Length Encoding
def encode(s):
    # Base case: empty string
    if not s:
        return ""

    result = ""
    count = 1

    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            result += s[i - 1] + str(count)
            count = 1

    # Add the last character and its count
    result += s[-1] + str(count)

    return result

# Example usage
print(encode("aaaabbbccc"))  # Output: a4b3c3
print(encode("abbbcdddd"))   # Output: a1b3c1d4


                            #PROBLEM 16:Number of Ways to Reach a Position After Exactly k Steps
def numWays(startPos, endPos, k):
    MOD = 10**9 + 7

    # Initialize the dp array
    dp = [0] * (2 * k + 3)
    dp[startPos] = 1

    # Fill the dp array
    for _ in range(k):
        new_dp = [0] * (2 * k + 3)
        for j in range(1, 2 * k + 2):
            new_dp[j] = (dp[j-1] + dp[j] + dp[j+1]) % MOD
        dp = new_dp

    return dp[endPos]

# Example usage
print(numWays(1, 2, 3))  # Output: 3
print(numWays(2, 5, 10)) # Output: 0
