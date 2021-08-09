#hello this is me doing some list comprehension practice

def fibo(n):
    return n if n<= 1 else (fibo(n-1) + fibo(n-2))
nums = [1,2,3,4,5,6]
#print([fibo(x) for x in nums])

words = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
#even_days = [x for x in words if len(x) % 2 == 0]
#print(even_days) # prints: ['Monday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# print("HELLO".isupper()) #returns true because in all caps
# print("hello".isupper()) #returns false because not in all caps
# print("Hello".isupper()) #returns false because not in all caps

words = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
make_lower = [d.lower() for d in words if d[0].isupper()] # only checking whether the first letter is uppercase or not to avoid the common error
#print(make_lower)

print([s for s in words if s.isupper() elif len(s) == 7(s)])

#This is an adaptation of one of Downey's examples in his book
def only_lower(t):
    res = []
    for s in t:
        if s.islower():
            res.append(s)
    return res

def only_lower(t):
    return [s for s in t if s.islower()]

