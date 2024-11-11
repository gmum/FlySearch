direct_naive = scan("direct_naive.txt")
direct_full = scan("direct_full.txt")

relative_naive = scan("relative_naive.txt")
relative_full = scan("relative_full.txt")

print("Direct Naive summary")
print(summary(direct_naive))

print("Direct Full summary")
print(summary(direct_full))

print("Relative Naive summary")
print(summary(relative_naive))

print("Relative Full summary")
print(summary(relative_full))

print("Wilcox test for direct")
print(wilcox.test(direct_naive, direct_full))

print("Wilcox test for relative")
print(wilcox.test(relative_naive, relative_full))