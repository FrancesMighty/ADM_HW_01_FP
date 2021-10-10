# ----------- INTRODUCTION -----------
# 01 --> Hello World-Python3
print("Hello, World!")
	
#02 --> If-else-Python3
#!/bin/python3

import math
import os
import random
import re
import sys

if __name__ == '__main__':
    n = int(input().strip())
    if n%2 != 0 or 6 <= n <= 20:
        print("Weird")
    else:
        print("Not Weird")

#03 --> Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    sum_ab = a + b
    diff_ab = a - b
    mul_ab = a * b
    print(sum_ab)
    print(diff_ab)  
    print(mul_ab)
	
#04 --> Division-Python3
from __future__ import division
# In Python2 you need the statement __future__ statement,without it both operators would return int value
# here I used Python3
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    i_div = a // b
    f_div = a / b
    print(i_div)
    print(f_div)
	
#05 --> Loops-Python3
if __name__ == '__main__':
    n = int(input())
    for i in range(n):
        print(i**2)

#06 --> Write a function-Python3
def is_leap(year):
    leap = False
    if year % 4 == 0 and year % 100 == 0 and year % 400 == 0:
        return True
    if year % 4 == 0 and year % 100 != 0:
        return True
    return leap

year = int(input())
print(is_leap(year))

#07 --> Print function-Python3
if __name__ == '__main__':
    n = int(input())
    num = ""
    for n in range(1, n + 1, 1):
        num = num + str(n)
    print(num)

# ----------- DATA TYPES -----------
# 01 --> List Comprehensions-Python3
if __name__ == '__main__':
    if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    r = [[i, j, k] for i in range(x + 1) for j in range(y + 1) for k in range(z + 1) if i + j + k != n]
    print(r)

#02 --> Runner-up Score-Python3
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    scores = list(set(arr))
    if len(scores) > 1:
        scores.remove(max(scores))
        run_up = max(scores)
    else:
        run_up = scores[0]
    print(run_up)

#03 -->	Nested List-Python3
if __name__ == '__main__':
    studs = dict()
    for _ in range(int(input())):
        name = input()
        score = float(input())
        if score in studs:
            studs[score] = studs[score]+ [name]
        else:
            studs[score] = [name]
    grades = list(studs.keys())
    min_g = grades.remove(min(grades))
    sl = min(grades)
    for name in sorted(studs[sl]):
        print(name)
		
#04 --> Percentage-Python3
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *s_marks = input().split()
        marks = list(map(float, s_marks))
        student_marks[name] = marks
    query_name = input()
    q_marks = student_marks[query_name]
    avarege = sum(q_marks)/len(q_marks)
    print("{:0.2f}".format(avarege))
	
#05 --> Lists-Python3
if __name__ == '__main__':
    l = list()
    N = int(input())
    for _ in range(N):
        fun_name, *args = input().split()
        args = list(map(float, args))
        if fun_name == "print":
            print(l)
        elif len(args) == 0:
            getattr(l, fun_name)()
        elif len(args) == 1:
            getattr(l, fun_name)(int(args[0]))
        else:
            getattr(l, fun_name)(int(args[0]), int(args[1]))
			
#06 --> Tuples-PyPy3 Pytho3 not available
if __name__ == '__main__':
    n = int(input())
    tuples = tuple(map(int, input().split()))
    print(hash(tuples))
	
# ----------- STRINGS -----------
# 01 --> SwapCases-Python3	
def swap_case(s):
    return s.swapcase()

if __name__ == '__main__':
    s = input()
    result = swap_case(s)
    print(result)

# 02 --> Split and Join-Python3		
def split_and_join(line):
    return "-".join(line.split())

if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)

# 03 --> What's your name?-Python3		
def print_full_name(first, last):
    # Write your code here
    print("Hello {} {}! You just delved into python.".format(first, last))

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)

# 04 --> Mutations-Python3
def mutate_string(string, position, character):
    return string[0:position] + character + string[position + 1::]

if __name__ == '__main__':
    s = input()
    i, c = input().split()
    s_new = mutate_string(s, int(i), c)
    print(s_new)
	
# 05 --> Find a string-Python3
def count_substring(string, sub_string):
    l = len(sub_string)
    counter = 0
    for i in range(0, len(string) - l + 1):
        if string[i: i + l] == sub_string:
            counter += 1
    return counter

if __name__ == '__main__':
    string = input().strip()
    sub_string = input().strip()
    
    count = count_substring(string, sub_string)
    print(count)
	
# 06 --> String validators-Python3
if __name__ == '__main__':
    s = input()
    for fun_name in ('isalnum', 'isalpha', 'isdigit', 'islower', 'isupper'):
        print(any(getattr(c, fun_name)() for c in s))
		
# 07 --> Text Alignment-Python3
#Replace all ______ with rjust, ljust or center. 
thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))
#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# 08 --> Text Wrap-Python3
import textwrap

def wrap(string, max_width):
    r = ""
    for i in range(0, len(string), max_width):
        r += string[i:i+ max_width] + "\n"
    return r

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)

# 09 --> Designer Door mat-Python3
N, M = map(int, input().split())
for i in range(1, N, 2):
    print(str('.|.' * i).center(M, '-'))
print('WELCOME'.center(M, '-'))
for i in range(N-2, -1, -2):
    print(str('.|.' * i).center(M, '-'))

# 10 --> String formatting-Python3
def print_formatted(number):
    lbin = len(bin(number)[2:])
    for i in range(1, number + 1, 1):
        octal = oct(i)
        hexa = hex(i)
        bina = bin(i)
        print(str(i).rjust(lbin,' '),end=" ")
        print(oct(i)[2:].rjust(lbin,' '),end=" ")
        print(((hex(i)[2:]).upper()).rjust(lbin,' '),end=" ")
        print(bin(i)[2:].rjust(lbin,' '),end=" ")
        print("")
        

if __name__ == '__main__':
    n = int(input())
    print_formatted(n)

# 11 --> Alphabet Rangoli-Python3	
def print_rangoli(size):
    all_letters = "abcdefghijklmnopqrstuvwxyz"
    rangoli = []
    for i in range(size):
        cur_l = "-".join(all_letters[i:size])
        rangoli.append((cur_l[::-1] + cur_l[1:]).center(4 * size - 3, "-"))
    print('\n'.join(rangoli[:0:-1] + rangoli))
    
if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)
	
# 12 --> Capitalize!-Python3
#!/bin/python3

import math
import os
import random
import re
import sys

# Complete the solve function below.
def solve(s):
    for i in s.split():
        s = s.replace(i,i.capitalize())
    return s

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()

# 13 --> Minion Game-Python3
def minion_game(s):
    points ={"Kevin" : 0, "Stuart" : 0}
    length = len(s)
    for i in range(length):
        if s[i] in "AEIOU":
            points["Kevin"] += (length)-i
        else :
            points["Stuart"] += (length)-i
    if points["Kevin"] == points["Stuart"]:
        print("Draw")
    else:
        winner = max(points, key = points.get)
        print(winner, points[winner])

if __name__ == '__main__':
    s = input()
    minion_game(s)

# 14 --> Merge the Tools-Python3
def merge_the_tools(string, k):
    n = len(string)
    for i in range(0, n - k + 1, k):
        ss = string[i:i + k]
        no_rep_ss = []
        for c in ss:
            if c not in no_rep_ss:
                no_rep_ss.append(c)
        print("".join(no_rep_ss))
    
if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)
	
	
# ----------- SETS -----------
# 01 --> Intro-Python3
def average(array):
    array = list(set(array))
    return "{:0.3f}".format(sum(array)/len(array))

if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
	
#02 --> No Idea-Python3
m_n = input().split()
lista = list(map(int, input().strip().split()))
m, n = int(m_n[0]), int(m_n[1])
happiness = 0
A = set(map(int, input().strip().split()))
B = set(map(int, input().strip().split()))

for i in lista:
    if i in A:
        happiness += 1
    if i in B:
        happiness -= 1
print(happiness)

#03 --> Symmetric difference-Python3
m = int(input())
M = set(map(int, input().split()))
n = int(input())
N = set(map(int, input().split()))
for el in (sorted(M.union(N) - M.intersection(N))):
    print(el)
	
#04 --> Set add-Python3
n = int(input())
S = set()
for _ in range(n):
    S.add(input())
print(len(S))

#05 --> Set discard ...-Python3
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for _ in range(N):
    fun_name, *args = input().split()
    args = list(map(float, args))
    if len(args) == 0:
        getattr(s, fun_name)()
    else:
        getattr(s, fun_name)(int(args[0]))
print(sum(s))

#06 --> Set union-Python3
n = int(input())
E = set(map(int, input().split()))
m = int(input())
F = set(map(int, input().split()))
print(len(F.union(E)))

#07 --> Set intersection-Python3
n = int(input())
E = set(map(int, input().split()))
m = int(input())
F = set(map(int, input().split()))
print(len(F.intersection(E)))

#08 --> Set difference-Python3
n = int(input())
E = set(map(int, input().split()))
m = int(input())
F = set(map(int, input().split()))
print(len(E.difference(F)))

#09 --> Set symmetryc diff-Python3
n = int(input())
E = set(map(int, input().split()))
m = int(input())
F = set(map(int, input().split()))
print(len(F.symmetric_difference(E)))

#10 --> Set mutations-Python3
n = int(input())
S = set(map(int, input().split()))
operations = int(input())
for _ in range(operations):
    fun_name, *len_s1 = input().split()
    s1 = set(map(int, input().split()))
    getattr(S, fun_name)(s1)
print(sum(S))

#11 --> Captain's Room-Python3
from collections import Counter
K = int(input())
rooms = Counter((map(int, input().split())))
for k in rooms:
    if rooms[k] == 1:
        print(k)
        break
#12 --> Check subset-Python3
T = int(input())
for n in range(T):
    lenA = int(input())
    A = set(map(int, input().split()))
    lenB = int(input())
    B = set(map(int, input().split()))
    print(A.issubset(B))

#13 --> Check strict subset-Python3
A = set(map(int, input().split()))
n = int(input())
bool = True
for _ in range(n):
    s = set(map(int, input().split()))
    if A.issuperset(s) and len(A) > len(s):
        bool = bool and True
    else:
        bool = bool and False
print(bool)

# ----------- COLLECTIONS -----------
# 01 --> Counter-Python3
from collections import Counter
n = int(input())
shoes_numbers = list(map(int, input().split()))
shoes = dict(Counter(shoes_numbers))
clients = int(input())
earning = 0
for _ in range(clients):
    info = list(map(int, input().split()))
    num, price = info[0], info[1]
    if num in shoes and shoes[num] > 0:
        earning += price
        shoes[num] = shoes[num] - 1
print(earning)

# 02 --> DefaultDict Tutorial-Python3
from collections import defaultdict
dd = defaultdict(list)
info = list(map(int, input().split()))
n, m = info[0], info[1]
for _ in range(n):
    w = input()
    dd[w].append(_ + 1)
r = []
for j in range(m):
    w = input()
    if w in dd.keys():
        r.append(" ".join([str(i) for i in dd[w]]))
    else:
        r.append(-1)
for i in r:
    print(i)
	
# 03 --> Collections.namedtuple()-Python3
from collections import namedtuple

num = int(input())
attrs = " ".join([name.replace(" ", "") for name in input().split()])
Student = namedtuple('Student', attrs)
students = []
for _ in range(num):
    infos = [name.replace(" ", "") for name in input().split()]
    students.append(Student(infos[0], infos[1], infos[2], infos[3]))
print(sum(int(i.MARKS) for i in students)/num)

# 04 --> Ordered Dict-Python3
from collections import OrderedDict
num = int(input())
d = OrderedDict()
for _ in range(num):
    info = input().split()
    name, price = " ".join(info[:-1]), int(info[-1])
    if name not in d.keys():
        d[name] = price
    else:
        d[name] = d[name] + price
for key, value in d.items():
    print(key, str(value))

# 05 --> Word Order-Python3
n = int(input())
w_n = dict()
for i in range(n):
    w = input()
    if w not in w_n.keys():
        w_n[w] = 1
    else:
        w_n[w] = w_n[w] + 1
print(len(w_n))
print(" ".join([str(w_n[x]) for x in w_n.keys()]))

#06 --> Deque-Python3
from collections import deque
d = deque()
n = int(input())
for _ in range(n):
    cmd = input().split()
    if len(cmd) == 1:
        getattr(d, cmd[0])()
    else:
        getattr(d, cmd[0])(eval(cmd[1]))
print(" ".join([str(el) for el in d]))

#07 --> Company Logo-Python3
#!/bin/python3
import math
import os
import random
import re
import sys

from collections import Counter
if __name__ == '__main__':
    s = Counter(sorted(input()))
    for i in s.most_common(3):
        print(i[0], i[1])

#08 --> Piling Up-Python3
from collections import deque

tests = int(input())
for _ in range(tests):
    boolean = True
    n = int(input())
    d = deque(map(int, input().split()))
    if(d[0] >= d[-1]):
        max = d.popleft()
    else:
        max = d.pop()
    while d:
        if(len(d)==1):
            if(d[0] <= max):
                break
            else:
                boolean = False
                break
        else:
            if(d[0]<=max and d[-1]<=max):
                if(d[0]>=d[-1]):
                    max = d.popleft()
                else:
                    max = d.pop()
            elif(d[0]<=max):
                max = d.popleft()
            elif(d[-1]<=max):
                max = d.pop()
            else:
                boolean = False
                break
    print("Yes" if boolean else "No")
	
# ----------- Date & Time -----------
# 01 --> Calendar module-Python3
import calendar, datetime
data = list((map(int, input().split())))
month, day, year = data[0], data[1], data[2]
my_date = datetime.date(year, month, day)
print(calendar.day_name[my_date.weekday()].upper())

# 02 --> Time Delta-Python3
#!/bin/python3
from datetime import datetime
T = int(input())
for _ in range(T):
    t1 = '%a %d %b %Y %H:%M:%S %z'
    t2 = '%a %d %b %Y %H:%M:%S %z'
    diff = abs(datetime.strptime(input(), t2) - datetime.strptime(input(), t1))
    print(int(diff.total_seconds()))

# ----------- Exceptions -----------
# 01 --> Exceptions-Python3
t = int(input())
for _ in range(t):
    try:
        values = list(map(int, input().split()))
        a, b = values[0], values[1]
        print(a//b)
    except Exception as e:
        print("Error Code:",e)
		
# ----------- Built-ins -----------
# 01 --> Zipped-Python3
values = list(map(int, input().split()))
N, X = values[0], values[1]
marks = []
for _ in range(X):
    marks.append(list(map(float, input().split())))
for m in zip(*marks):
    print("{:0.1f}".format(sum(m)/X))
	
# 02 --> Athlete Sort-Python3
#!/bin/python3
import math
import os
import random
import re
import sys

if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    arr = sorted(arr, key=lambda x: x[k])
    arr = map(lambda x: [str(x[i]) for i in range(len(x))],arr)
    for el in arr:
        print(" ".join(el))
		
#03 --> ginortS-Python3
s = input()
r = sorted(s, key=lambda c: (c.isdigit() - c.islower(), c in '02468', c))
print("".join(r))


# ----------- Functionals -----------
#01 --> Map and Lambda function-Python3
cube = lambda x: x**3 

def fibonacci(n):
    l = []
    prec, cur = 0, 1
    for _ in range(n):
        if _ == 0:
            l.append(0)
        elif _ == 1:
            l.append(1)
        else:
            next_n = prec + cur
            l.append(next_n)
            prec = cur
            cur = next_n
    return l   

if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))


# ----------- Regex & parsing -----------
#01 --> Detect Floating Point Number-Python3
import re
t = int(input())
for _ in range(t):
    s = input()
    pattern = re.compile("^[+-]?([0-9]*\.)[0-9]+$") 
    if pattern.match(s):
        print(True)
    else:
        print(False)
		
#02 --> Re.split()-Python3
regex_pattern = r"[,.]"	

import re
print("\n".join(re.split(regex_pattern, input())))

#03 --> Group(), Groups() & Groupdict()-Python3
import re
s = input()
m = re.search(r'([A-Za-z0-9])\1+',s)
print(m.group(1) if m else -1)

#04 --> Re.findall() & re.finditer()-Python3
import re
s = input().strip()
st = re.findall(r'(?<=[qwrtypsdfghjklzxcvbnm])([aeiou]{2,})(?=[qwrtypsdfghjklzxcvbnm])', s, re.IGNORECASE)
if st:
    for el in st:
        print(el)
else:
    print(-1)
	
#05 --> Re.start() & re.end()-Python3
import re

s = input()
k = input()
ms = re.finditer(r'(?=(' + k + '))', s)
a = [(m.start(1), m.end(1) - 1) for m in ms]
if len(a) == 0:
    print((-1, -1))
else:
    print(*a, sep='\n')

#06 --> Re. substitution-Python3
import re

def substitute(m):
    return 'and' if m.group(1) == '&&' else 'or'
    
N = int(input())
for _ in range(N):
    s = input()
    print(re.sub(r"(?<= )(\|\||&&)(?= )", substitute, s))
	
	
#07 --> Validating Credit Card Numbers-Python3
import re
for i in range(int(input())):
    s = input()
    if not re.search(r'(\d)\1{3,}', s.replace("-", "")) and re.search(r'^[456]\d{3}(-?)\d{4}\1\d{4}\1\d{4}$', s):
        print('Valid')
    else:
        print("Invalid")
		
#08 --> Validating Postal Codes-Python3
regex_integer_in_range = r"^[1-9][0-9]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"([0-9])(?=[0-9]\1)"	# Do not delete 'r'.

import re
P = input()

print (bool(re.match(regex_integer_in_range, P)) 
and len(re.findall(regex_alternating_repetitive_digit_pair, P)) < 2)


#09 --> Matrix Scripts-Python3
import math
import os
import random
import re
import sys

first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []

for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
matrix = list(zip(*matrix))

app = str()
for words in matrix:
    app += "".join(char for char in words)
       
print(re.sub(r'(?<=\w)([^\w\d]+)(?=\w)', ' ', app))

# ----------- XML -----------
# 01 --> Find the Score-Python3	
import sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    count = 0
    for tag in node:
        count = count + get_attr_number(tag)
    return count + len(node.attrib)

if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# 02 --> Find the Score-Python3	
import xml.etree.ElementTree as etree

maxdepth = 0
def depth(elem, level):
    global maxdepth
    if (level == maxdepth):
        maxdepth += 1

    for child in elem:
        depth(child, level + 1)

if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)
	
# ----------- CLOSURES AND DECORATIONS -----------
# 01 --> Standardize Mobile Number Using Decorators-Python3	
def wrapper(f):
    def fun(l):
        f(['+91 ' + n[-10:-5] + ' ' + n[-5:] for n in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l)
	
# 02 --> Decorators 2 - Name Directory-Python3
import operator

def person_lister(f):
    def inner(people):
        return map(f, sorted(people, key=lambda x: int(x[2])))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')


# ----------- CLOSURES AND DECORATIONS -----------
#01 --> Arrays-Python3
import numpy

def arrays(arr):
    return(numpy.array(arr[::-1], float))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

#02 --> Shape and Reshape-Python3
import numpy
arr = input().split()
print(numpy.array(arr, int).reshape(3, 3))

#03 --> Transpose and Flatten-Python3
import numpy

n, m  = map(int, input().split())
arr = numpy.array([input().strip().split() for _ in range(n)], int)
print(arr.transpose())
print(arr.flatten())

#04 --> Concatenate-Python3
import numpy
P, N, M = map(int,input().split())
arr1 = numpy.array([input().split() for _ in range(P)],int)
arr2 = numpy.array([input().split() for _ in range(N)],int)
print(numpy.concatenate((arr1, arr2), axis = 0))

#05 --> Zeros and Ones-Python3
import numpy

N = tuple(map(int, input().split()))
print(numpy.zeros(N, int))
print(numpy.ones(N, int))


#06 --> Eye and Identity-Python3
import numpy

numpy.set_printoptions(legacy='1.13')
print(numpy.eye(*map(int, input().split())))

#07 --> Array Mathematics-Python3
import numpy

N, M = map(int, input().split())

A = numpy.array([list(map(int, input().split())) for n in range(N)])
B = numpy.array([list(map(int, input().split())) for n in range(N)])

print(A + B)
print(A - B)
print(A * B)
print(A // B)
print(A % B)
print(A ** B)


#08 --> Floor, ceil and rint-Python3
import numpy

numpy.set_printoptions(legacy='1.13')

array = numpy.array(input().split(),float)

print(numpy.floor(array))
print(numpy.ceil(array))
print(numpy.rint(array))

#09 --> Sum and Prod-Python3
import numpy

N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)], int)
print(numpy.prod(numpy.sum(A, axis=0), axis=0))

#10 --> Min and Max-Python3import numpy
import numpy

N, M = map(int, input().split())
A = numpy.array([input().split() for _ in range(N)],int)
print(numpy.max(numpy.min(A, axis=1), axis=0))

#11 --> Polynomials-Pyhton3
import numpy

polynomial = [float(x) for x in input().split()]
x = float(input())

print(numpy.polyval(polynomial, x))






	
	
	
	
	