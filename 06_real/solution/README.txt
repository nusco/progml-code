In the book, I ask you to guess which character in MNIST is harder for our
classifier to recognize. To find a precise answer, I modified the code to run
training and testing on each of the 10 digits.

Note that the code I ended up with is very inefficient: it runs 100 training
iterations on each digit. In the next chapter, you'll see a much more efficient
way to deal with multiple digits.

Here are the results of this experiment:

Correct classifications for digit 0: 9899/10000 (98.99%)
Correct classifications for digit 1: 9903/10000 (99.03%)
Correct classifications for digit 2: 9737/10000 (97.37%)
Correct classifications for digit 3: 9698/10000 (96.98%)
Correct classifications for digit 4: 9759/10000 (97.59%)
Correct classifications for digit 5: 9637/10000 (96.37%)
Correct classifications for digit 6: 9807/10000 (98.07%)
Correct classifications for digit 7: 9814/10000 (98.14%)
Correct classifications for digit 8: 9385/10000 (93.85%)
Correct classifications for digit 9: 9557/10000 (95.57%)

It seems that the digit 8 is giving a hard time to the classifier–more than any
other digit. On the other hand, 1s are particularly easy to classify.

For the record, my guess was that the classifier would struggle to recognize 7s
or 4s, because of all the different ways they can be written. I was quite wrong.

It would be interesting to try and understand why the digit 8 is hard to
classify. Is the classifier routinely mistaking 8s for other digits (what is
called a "false negative")? Or is it mostly mistaking other digits for 8s (a
"false positive")? Which other digits exactly? If you're looking for a
challenge, you can delve even deeper into the data, searching for answers to
those questions.
