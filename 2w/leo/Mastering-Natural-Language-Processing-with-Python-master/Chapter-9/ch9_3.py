import nltk
expr_read = nltk.sem.DrtExpression.from string
expr3 = expr_read('([x], [John(x), eats(x)])+ ([y],[Sam(y),eats(y)])')
print(expr3)
print(expr3.simplify())
expr3.draw()
