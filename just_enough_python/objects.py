from datetime import date
moon_landing = date(1969, 7, 20)

moon_landing.weekday()  # => 6

viking_1_mars_landing = moon_landing.replace(year=1976)
viking_1_mars_landing.strftime("%d/%m/%y")  # => 20/07/76

'strings are objects'.upper()  # => 'STRINGS ARE OBJECTS'
