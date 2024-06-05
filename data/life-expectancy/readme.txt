# Life Expectancy Data

Welcome to your first real-world data analysis! This dataset contains life expectancy at birth for most world countries.

I selected these data from the wonderful [Our World in Data](https://ourworldindata.org) site. All the data refers to the year 2014. I skipped the countries missing any data in that year. (That includes a few major countries).


## Data Format

The data are in `life-expectancy.txt`. It includes these columns:

* The name of the country that the example refers to.
* [Air Pollution](https://ourworldindata.org/air-pollution), in micrograms per cubic meter.
* [Healthcare Expenditures](https://ourworldindata.org/financing-healthcare) as a share of the national GDP.
* [Access to Improved Drinking Water](https://ourworldindata.org/water-access-resources-sanitation), such as piped household water or public taps, as a share of the total population.
* [Life Expectancy](https://ourworldindata.org/life-expectancy) at birth in years.

You can use the second, third, and fourth column as input variables, and the last column as the label. (But feel free to pick another column as the label if you wish!)

NumPy's arrays can only contain uniform data, so you cannot mix strings and floats in the same matrix. To make it easier for you to load the data, I created a second file named `life-expectancy-without-country-names.txt`. That file contains the same data as `life-expectancy.txt`, only without the first column. It has the same exact format as the `pizza_3_vars.txt` file that you've been using so far in the book, so it's a drop-in replacement. You can use the same code that we used on `pizza_3_vars.txt`, and just change the name of the data file in your code.


## A Few Things to Do

Here is one way to use these data. (But feel free to come up with your own experiments.)

1. Change the `multiple_regression_final.py` program to load `life-expectancy-without-country-names.txt` instead of `pizza_3_vars.txt`.
2. Find good hyperparameters to train the program. You might find that the program needs a very small learning rate to avoid generating "Not a Number" errors. With a small `lr`, you need more iterations. Don't be afraid to ask for millions of iterations if that helps you find a low loss.
3. Look at the predictions. Are they close to the labels? If they aren't, can you explain why?


## Mandatory Statistics Disclaimer

Don't roll your eyes–you _knew_ this was coming. ;)

I compiled these data under the assumption that there is a relation between, for example, access to drinkable water and life expectancy. However, that doesn't mean that less drinkable water necessarily shortens life expectancy. Even if the weights from your machine learning algorithm seem to indicate that A correlates with B, you should always resist the temptation to jump to the conclusion that A *causes* B. The causal arrow might be pointing in the other direction, and it's B that causes A. More commonly, there might be an unknown variable C that causes both A and B.

As statisticians are keen to repeat: "correlation doesn't imply causation". That's worth remembering every time you analyze data.

Also, if your sample size is small (as in this case, because there aren't that many nations in the world), then you should always consider the possibility that the correlation you find is accidental.


## Licence

The data from [Our World in Data](https://ourworldindata.org) is released under a [Creative Commons](https://creativecommons.org/licenses/by-sa/4.0/) license. The content of this directory, including this file and the `life-expectancy*.txt` files, is a transformation of those data, so it's also released under the same license.

Enjoy!
