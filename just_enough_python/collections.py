a_tuple = (3, 9, 12, 7, 1, -4)
len(a_tuple)  # => 6
a_tuple[2]    # => 12
a_tuple[2:5]  # => (12, 7, 1)

a_list = [10, 20, 30]
a_list[1] = a_list[1] + 2
a_list.append(100)
a_list  # => [10, 22, 30, 100]

a_mixed_list = ['a', 42, False, (10, 20), 99.9, a_list]
