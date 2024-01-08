                        #1401. Circle and Rectangle Overlapping

def checkOverlap(radius, xCenter, yCenter, x1, y1, x2, y2):
    # Calculate the closest point on the rectangle to the circle
    closestX = max(x1, min(xCenter, x2))
    closestY = max(y1, min(yCenter, y2))

    # Calculate the distance between the circle center and the closest point
    distance = ((xCenter - closestX) ** 2 + (yCenter - closestY) ** 2) ** 0.5

    # Check if the distance is less than or equal to the radius
    return distance <= radius

# Test cases
print(checkOverlap(1, 0, 0, 1, -1, 3, 1))  # Output: True
print(checkOverlap(1, 1, 1, 1, -3, 2, -1))  # Output: False
print(checkOverlap(1, 0, 0, -1, 0, 0, 1))   # Output: True


                        #1823. Find the Winner of the Circular Game

def findTheWinner(n, k):
    # Create a list to represent the circle of friends
    friends = list(range(1, n+1))

    # Start the game at friend 1
    current_index = 0

    # Repeat the game until there is only one friend left
    while len(friends) > 1:
        # Count k friends clockwise and determine the index of the friend to be removed
        current_index = (current_index + k - 1) % len(friends)
        
        # Remove the friend from the circle
        friends.pop(current_index)

    # The last remaining friend is the winner
    return friends[0]

# Test cases
print(findTheWinner(5, 2))  # Output: 3
print(findTheWinner(6, 5))  # Output: 1

                                  #354. Russian Doll Envelopes

def maxEnvelopes(envelopes):
    if not envelopes:
        return 0

    # Sort the envelopes first by width and then by height
    envelopes.sort(key=lambda x: (x[0], -x[1]))

    # Helper function for binary search to find the insertion point
    def binary_search(target, dp_len):
        left, right = 0, dp_len - 1
        while left <= right:
            mid = (left + right) // 2
            if dp[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left

    # Initialize an array to store the height values of the envelopes
    dp = [envelopes[0][1]]

    # Iterate through the envelopes and update the dp array
    for i in range(1, len(envelopes)):
        height = envelopes[i][1]

        if height > dp[-1]:
            dp.append(height)
        else:
            # Use binary search to find the insertion point for the current height
            insert_index = binary_search(height, len(dp))
            dp[insert_index] = height

    # The length of the dp array is the maximum number of envelopes that can be Russian doll
    return len(dp)

# Test case
envelopes1 = [[5,4],[6,4],[6,7],[2,3]]
print(maxEnvelopes(envelopes1))  # Output: 3

envelopes2 = [[1,1],[1,1],[1,1]]
print(maxEnvelopes(envelopes2))  # Output: 1


                                #661.Image Smoother

def imageSmoother(img):
    m, n = len(img), len(img[0])
    result = [[0] * n for _ in range(m)]

    for i in range(m):
        for j in range(n):
            count = 0
            total_sum = 0

            # Iterate over the surrounding cells
            for x in range(max(0, i-1), min(m, i+2)):
                for y in range(max(0, j-1), min(n, j+2)):
                    total_sum += img[x][y]
                    count += 1

            result[i][j] = total_sum // count

    return result

# Test cases
img1 = [[1,1,1],[1,0,1],[1,1,1]]
print(imageSmoother(img1))
# Output: [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

img2 = [[100,200,100],[200,50,200],[100,200,100]]
print(imageSmoother(img2))
# Output: [[137, 141, 137], [141, 138, 141], [137, 141, 137]]


                                #462. Minimum Moves to Equal Array Elements II

def minMoves2(nums):
    nums.sort()
    median = nums[len(nums) // 2]
    
    # Calculate the sum of absolute differences between each element and the median
    moves = sum(abs(num - median) for num in nums)
    
    return moves

# Test cases
nums1 = [1,2,3]
print(minMoves2(nums1))  # Output: 2

nums2 = [1,10,2,9]
print(minMoves2(nums2))  # Output: 16


                            #497. Random Point in Non-overlapping Rectangles

import random

class Solution:
    def __init__(self, rects):
        self.rects = rects
        self.total_area = sum((x2 - x1 + 1) * (y2 - y1 + 1) for x1, y1, x2, y2 in rects)
        self.cumulative_areas = [0]

        # Calculate cumulative areas
        for x1, y1, x2, y2 in rects:
            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            self.cumulative_areas.append(self.cumulative_areas[-1] + area)

    def pick(self):
        # Choose a random integer area uniformly from [1, total_area]
        target_area = random.randint(1, self.total_area)

        # Find the corresponding rectangle
        for i, cumulative_area in enumerate(self.cumulative_areas):
            if target_area <= cumulative_area:
                x1, y1, x2, y2 = self.rects[i - 1]
                width = x2 - x1 + 1
                height = y2 - y1 + 1

                # Calculate the coordinates of the random point within the rectangle
                offset = target_area - self.cumulative_areas[i - 1]
                row = offset // width
                col = offset % width

                return [x1 + col, y1 + row]

# Test case
solution = Solution([[-2, -2, 1, 1], [2, 2, 4, 6]])
print(solution.pick())  # Output: [1, -2]
print(solution.pick())  # Output: [1, -1]
print(solution.pick())  # Output: [-1, -2]
print(solution.pick())  # Output: [-2, -2]
print(solution.pick())  # Output: [0, 0]


                                 #299. Bulls and Cows

def getHint(secret, guess):
    bulls, cows = 0, 0
    secret_count = {}
    guess_count = {}

    for s, g in zip(secret, guess):
        if s == g:
            bulls += 1
        else:
            # Update counts for non-bull digits in secret and guess
            secret_count[s] = secret_count.get(s, 0) + 1
            guess_count[g] = guess_count.get(g, 0) + 1

    # Count cows by finding common digits in secret and guess
    for digit, count in guess_count.items():
        if digit in secret_count:
            cows += min(count, secret_count[digit])

    return f"{bulls}A{cows}B"

# Test cases
print(getHint("1807", "7810"))  # Output: "1A3B"
print(getHint("1123", "0111"))  # Output: "1A1B"


                                   #1248. Count Number of Nice Subarrays

def numberOfSubarrays(nums, k):
    def atMost(nums, k):
        count = i = 0
        result = 0

        for j in range(len(nums)):
            if nums[j] % 2 != 0:
                k -= 1

            while k < 0:
                if nums[i] % 2 != 0:
                    k += 1
                i += 1

            count += j - i + 1
            result += count

        return result

    return atMost(nums, k) - atMost(nums, k - 1)

# Test cases
print(numberOfSubarrays([1,1,2,1,1], 3))  # Output: 2
print(numberOfSubarrays([2,4,6], 1))        # Output: 0
print(numberOfSubarrays([2,2,2,1,2,2,1,2,2,2], 2))  # Output: 16


                             #187. Repeated DNA Sequences

def findRepeatedDnaSequences(s):
    seen = set()
    result = set()

    for i in range(len(s) - 9):
        sequence = s[i:i+10]
        if sequence in seen:
            result.add(sequence)
        else:
            seen.add(sequence)

    return list(result)

# Test case
s1 = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
print(findRepeatedDnaSequences(s1))  # Output: ["AAAAACCCCC","CCCCCAAAAA"]

s2 = "AAAAAAAAAAAAA"
print(findRepeatedDnaSequences(s2))  # Output: ["AAAAAAAAAA"]


                #1334. Find the City With the Smallest Number of Neighbors at a Threshold Distance

def findTheCity(n, edges, distanceThreshold):
    # Initialize the distance matrix with a large value for unreachable pairs
    dist = [[float('inf')] * n for _ in range(n)]

    # Set the initial distances based on the given edges
    for from_city, to_city, weight in edges:
        dist[from_city][to_city] = weight
        dist[to_city][from_city] = weight

    # Update the distance matrix using the Floyd-Warshall algorithm
    for k in range(n):
        for i in range(n):
            for j in range(n):
                dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])

    min_neighbors = float('inf')
    result_city = -1

    # Iterate through each city and count the number of reachable cities
    for i in range(n):
        neighbors = sum(1 for d in dist[i] if d <= distanceThreshold)
        if neighbors <= min_neighbors:
            min_neighbors = neighbors
            result_city = i

    return result_city

# Test case
print(findTheCity(4, [[0,1,3],[1,2,1],[1,3,4],[2,3,1]], 4))  # Output: 3
print(findTheCity(5, [[0,1,2],[0,4,8],[1,2,3],[1,4,2],[2,3,1],[3,4,1]], 2))  # Output: 0


                             #2970. Count the Number of Incremovable Subarrays

def countIncremovableSubarrays(nums):
    n = len(nums)
    count = 0
    dp = [0] * n  # dp[i] represents the number of incremovable subarrays ending at index i

    for i in range(1, n):
        if nums[i] > nums[i - 1]:
            dp[i] = dp[i - 1] + 1
            count += dp[i]

    return count

# Test cases
print(countIncremovableSubarrays([1,2,3,4]))  # Output: 10
print(countIncremovableSubarrays([6,5,7,8]))  # Output: 7
print(countIncremovableSubarrays([8,7,6,6]))  # Output: 3


                    #2002. Maximum Product of the Length of Two Palindromic Subsequences

def maxProduct(s):
    n = len(s)

    def isPalindrome(mask):
        subset = ""
        for i in range(n):
            if (mask >> i) & 1:
                subset += s[i]
        return subset == subset[::-1]

    dp = [0] * (1 << n)
    for mask in range(1, 1 << n):
        for i in range(n):
            if (mask >> i) & 1:
                dp[mask] = max(dp[mask], len(s[i]) + dp[mask ^ (1 << i)])

    maxProduct = 0
    for mask in range(1, 1 << n):
        if isPalindrome(mask):
            maxProduct = max(maxProduct, dp[mask] * dp[((1 << n) - 1) ^ mask])

    return maxProduct

# Test cases
print(maxProduct("leetcodecom"))  # Output: 9
print(maxProduct("bb"))           # Output: 1
print(maxProduct("accbcaxxcxx"))  # Output: 25


                           #324. Wiggle Sort II

def wiggleSort(nums):
    n = len(nums)
    nums.sort()

    mid = n // 2
    smaller_half = nums[:mid][::-1]
    larger_half = nums[mid:][::-1]

    for i in range(mid):
        nums[2 * i] = smaller_half[i]

    for i in range(mid, n):
        nums[2 * (i - mid) + 1] = larger_half[i - mid]

# Test case
nums1 = [1, 5, 1, 1, 6, 4]
wiggleSort(nums1)
print(nums1)  # Output: [1, 6, 1, 5, 1, 4]

nums2 = [1, 3, 2, 2, 3, 1]
wiggleSort(nums2)
print(nums2)  # Output: [2, 3, 1, 3, 1, 2]


                             #638. Shopping Offers

def shopping_offers(price, special, needs):
    memo = {}

    def dp(needs):
        if tuple(needs) in memo:
            return memo[tuple(needs)]

        result = sum(needs[i] * price[i] for i in range(len(needs)))

        for offer in special:
            new_needs = [needs[j] - offer[j] for j in range(len(needs))]
            if all(x >= 0 for x in new_needs):
                result = min(result, offer[-1] + dp(new_needs))

        memo[tuple(needs)] = result
        return result

    return dp(needs)

# Test case 1
price1 = [2, 5]
special1 = [[3, 0, 5], [1, 2, 10]]
needs1 = [3, 2]
output1 = shopping_offers(price1, special1, needs1)
print(output1)  # Output: 14

# Test case 2
price2 = [2, 3, 4]
special2 = [[1, 1, 0, 4], [2, 2, 1, 9]]
needs2 = [1, 2, 1]
output2 = shopping_offers(price2, special2, needs2)
print(output2)  # Output: 11


                               #2976. Minimum Cost to Convert String I

import heapq

def minimumCost(source, target, original, changed, cost):
    n = len(source)

    # Build a graph
    graph = {}
    for i in range(len(original)):
        x, y, z = original[i], changed[i], cost[i]
        if x not in graph:
            graph[x] = {}
        graph[x][y] = min(graph[x].get(y, float('inf')), z)

    # Dijkstra's algorithm
    pq = [(0, source)]
    visited = set()

    while pq:
        cur_cost, cur_source = heapq.heappop(pq)

        if cur_source == target:
            return cur_cost

        if cur_source in visited:
            continue
        visited.add(cur_source)

        for i in range(n):
            x, y = cur_source[i], target[i]
            if x != y and x in graph and y in graph[x]:
                heapq.heappush(pq, (cur_cost + graph[x][y], cur_source[:i] + y + cur_source[i+1:]))

    return -1

# Test case 1
source1 = "abcd"
target1 = "acbe"
original1 = ["a", "b", "c", "c", "e", "d"]
changed1 = ["b", "c", "b", "e", "b", "e"]
cost1 = [2, 5, 5, 1, 2, 20]
output1 = minimumCost(source1, target1, original1, changed1, cost1)
print(output1)  # Output: 28

# Test case 2
source2 = "aaaa"
target2 = "bbbb"
original2 = ["a", "c"]
changed2 = ["c", "b"]
cost2 = [1, 2]
output2 = minimumCost(source2, target2, original2, changed2, cost2)
print(output2)  # Output: 12

# Test case 3
source3 = "abcd"
target3 = "abce"
original3 = ["a"]
changed3 = ["e"]
cost3 = [10000]
output3 = minimumCost(source3, target3, original3, changed3, cost3)
print(output3)  # Output: -1


