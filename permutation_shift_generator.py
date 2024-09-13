class PermutationShiftGenerator:
    def __init__(self, arr: list):
        self.arr = arr
        self.doubled_arr = arr + arr
        self.length = len(arr)

    def __iter__(self):
        def iterator_function():
            for i in range(self.length):
                yield self.doubled_arr[i:i + self.length]

        return iterator_function()


def main():
    arr = [1, 2, 3, 4]
    sg1 = PermutationShiftGenerator(arr)

    for i in sg1:
        print(i)

    print("===")

    arr = []
    sg2 = PermutationShiftGenerator(arr)

    for i in sg2:
        print(i)

    print("===")

    arr = [1]
    sg3 = PermutationShiftGenerator(arr)

    for i in sg3:
        print(i)


if __name__ == "__main__":
    main()
