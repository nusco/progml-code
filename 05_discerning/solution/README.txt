Here are the four weights at the end of training:

- Weight of the bias column: -0.37450392
- Weight of the Reservations column: 0.51754011
- Weight of the Temperature column: -0.35263466
- Weight of the Tourists column: 0.25625742

Think about what these weights mean. The final prediction of the classifier is
the sum of the weights above, each multiplied by the value in the matching
column, and finally passed through a sigmoid:

sigmoid(1 * weight_of_bias +
        reservations * weight_of_reservations +
        temperature * weight_of_temperature +
        tourists * weight_of_tourists)

That means that the larger the weight, the more impact the matching input
variable has on the final result.

Note that the final result also depends on the values in each column, so we
cannot just compare the weights–we also need to know how big the input values
are, and whether they're positive or negative. In our case, however, all the
columns contain values that are positive, and in a roughly similar range (from
0 to a few dozens). Because the inputs are pretty uniform, we can look at the
relative size of the weights as a hint of how much each column impacts the
final result.

That's far from a statistical analysis, but we can use those consideration as a
heuristic to guess a few interesting facts:

1. The reservations seem to have a large effect on the final result. As we
expected, more reservations generally result in a higher chance of a neighbor
calling the police.

2. The temperature seems to have a smaller impact, and it also has a negative
weight. That means that higher temperatures tend to result in a slightly lower
chance to get a call to the police. Maybe people tend to tolerate noise better
in summer, possibly because many of them are out of town on vacation.

3. The weight associated with the Tourists column is small and positive. Also,
note that the values in that column tend to be a bit smaller than the numbers
in the other columns. It's hard to derive any conclusion from those facts, but
we can get a hint: the products of the small inputs with this small weight are
probably themselves small, which means that the number of tourists doesn't have
a great impact on the final result.
