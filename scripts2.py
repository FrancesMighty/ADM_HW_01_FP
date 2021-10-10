# 01 --> BIRTHDAY CAKE CANDLES-PYTHON3
!/bin/python3

import math
import os
import random
import re
import sys

def birthdayCakeCandles(candles):
    m = max(candles)
    return candles.count(m)

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()
	
# 02 --> KANGAROO-PYTHON3
def kangaroo(x1, v1, x2, v2):
    # Write your code here
    if (x1 == x2 and v1 != v2) or (x1 < x2 and v1 <= v2) or (x1 > x2 and v1 >= v2):
        return "NO"
    while x1 > x2:
        x1 += v1
        x2 += v2
    if x1 == x2:
        return "YES"
    while x1 < x2:
        x1 += v1
        x2 += v2
    if x1 == x2:
        return "YES"
    return "NO"
        
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
	
# 03 --> VIRAL ADVERTISING-PYTHON3	
#!/bin/python3

import math
import os
import random
import re
import sys

def viralAdvertising(n):
    shared = 5
    tot = 0
    for _ in range(n):
        like = shared//2
        tot += like
        shared = like * 3
    return tot

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    n = int(input().strip())

    result = viralAdvertising(n)

    fptr.write(str(result) + '\n')

    fptr.close()
	
# 04 --> RECURSIVE DIGIT SUM-PYTHON3
#!/bin/python3

import math
import os
import random
import re
import sys

def superDigit(n, k):
    x = int(n) * int(k) % 9
    return x if x else 9
    

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    n = first_multiple_input[0]

    k = int(first_multiple_input[1])

    result = superDigit(n, k)

    fptr.write(str(result) + '\n')

    fptr.close()
	
# 05 --> INSERTION SORT PART1-PYTHON3
#!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort1(n, arr):
    N = arr[n-1]
    i = n - 2
    while i >= 0 and arr[i] > N:
        arr[i + 1] = arr[i]
        print(*arr)
        i -= 1
    arr[i + 1] = N
    print(*arr)
    

if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort1(n, arr)


# 06 --> INSERTION SORT PART2-PYTHON3	
#!/bin/python3

import math
import os
import random
import re
import sys

def insertionSort1(i, arr):
    j = i;
    while (j > 0 and arr[j] < arr[j-1]):
        curr = arr[j];
        arr[j] = arr[j-1];
        arr[j-1] = curr;
        j -= 1;
    if i !=0:
        print(*arr)
    
def insertionSort2(n, arr):   
    for i in range(n):
        j = i;
        insertionSort1(i, arr)
        
if __name__ == '__main__':
    n = int(input().strip())

    arr = list(map(int, input().rstrip().split()))

    insertionSort2(n, arr)


	

	
	