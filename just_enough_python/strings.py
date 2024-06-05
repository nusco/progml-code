s1 = "This is a string"
s2 = 'this is also a string'
s1 + " and " + s2  # => 'This is a string and this is also a string'

print('Yup, this is yet another "Hello, World!" example')

s3 = s2[8:12]
s3  # => 'also'

a = 1
b = 99.12345
c = 'X'
"The values of these variables are %d, %.2f, and %s" % (a, b, c)
# => The values of these variables are 1, 99.12, and X

"Less than %.d%% of ML books have a hammer on the cover" % a
# => Less than 1% of ML books have a hammer on the cover
