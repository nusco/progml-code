def welcome(user):
    PASSWORD = "1234"
    message = "Hi, %s! Your new password is %s" % (user, PASSWORD)
    return message


welcome("Roberto")  # => 'Hi, Roberto! Your new password is 1234'


def modify_list(l):
    l.append(42)


a_list = [1, 2, 3]
modify_list(a_list)
a_list  # => [1, 2, 3, 42]


def welcome(user, secure):
    if secure:
        PASSWORD = "123456"
    else:
        PASSWORD = "1234"
    return "Hi, %s! Your new password is %s" % (user, PASSWORD)


welcome("Roberto", True)  # => Hi, Roberto! Your new password is 123456

welcome("Roberto", secure=True)  # => Hi, Roberto! Your new password is 123456

welcome(secure=False, user="Mike")  # => Hi, Mike! Your new password is 1234


def welcome(user="dear user", secure=True):
    if secure:
        PASSWORD = "123456"
    else:
        PASSWORD = "1234"
    return "Hi, %s! Your new password is %s" % (user, PASSWORD)


welcome()  # => Hi, dear user! Your new password is 123456

welcome(secure=False)  # => Hi, dear user! Your new password is 1234
