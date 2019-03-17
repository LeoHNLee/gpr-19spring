import nltk
expr_read = nltk.sem.DrtExpression.from string
expr2 = expr_read('([x,y], [John(x), Went(x),Sam(y),Meet(x,y)])')
print(expr2)
expr2.draw()
print(expr2.fol())
