direct_naive = scan("chisq_direct_naive.txt")
direct_full = scan("chisq_direct_full.txt")

relative_naive = scan("chisq_relative_naive.txt")
relative_full = scan("chisq_relative_full.txt")

print("Direct Naive summary")
print(summary(direct_naive))

print("Direct Full summary")
print(summary(direct_full))

print("Relative Naive summary")
print(summary(relative_naive))

print("Relative Full summary")
print(summary(relative_full))

direct_naive_ones = sum(direct_naive == 1)
direct_naive_zeros = sum(direct_naive == 0)

direct_full_ones = sum(direct_full == 1)
direct_full_zeros = sum(direct_full == 0)

print("Direct Naive ones")
print(direct_naive_ones)

print("Direct Naive zeros")
print(direct_naive_zeros)

print("Direct Full ones")
print(direct_full_ones)

print("Direct Full zeros")
print(direct_full_zeros)

direct_naive_contingency = c(direct_naive_zeros, direct_naive_ones)
direct_full_contingency = c(direct_full_zeros, direct_full_ones)

chisq.test(direct_naive_contingency, p=direct_full_contingency, rescale.p=TRUE)

relative_naive_ones = sum(relative_naive == 1)
relative_naive_zeros = sum(relative_naive == 0)

relative_full_ones = sum(relative_full == 1)
relative_full_zeros = sum(relative_full == 0)

print("Relative Naive ones")
print(relative_naive_ones)

print("Relative Naive zeros")
print(relative_naive_zeros)

print("Relative Full ones")
print(relative_full_ones)

print("Relative Full zeros")
print(relative_full_zeros)

relative_naive_contingency = c(relative_naive_zeros, relative_naive_ones)
relative_full_contingency = c(relative_full_zeros, relative_full_ones)

chisq.test(relative_naive_contingency, p=relative_full_contingency, rescale.p=TRUE)