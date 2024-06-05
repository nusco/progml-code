print("Hello, Python!")

x = "Hello!"
x = 42

x = 7
if x > 5:
    print("x is greater than 5")
    print("And while we're here...")
    if x < 10:
        print("...it's also smaller than 10")
else:
    print("x is less or equal than 5")

an_integer = 42
a_float = 0.5
a_boolean = True
a_string = "abc"

an_integer + a_float   # => 42.5
an_integer >= a_float  # => True
a_float * 2            # => 1.0
an_integer / 10        # => 4.2
an_integer % 10        # => 2
not a_boolean          # => False
a_boolean or False     # => True
a_string + 'def'       # => 'abcdef'

"20" + str(20)  # => '2020'
