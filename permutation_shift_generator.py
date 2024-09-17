class PermutationShiftGenerator:
    def __init__(self, arr: list, dummy: bool = False):
        self.arr = arr
        self.doubled_arr = arr + arr
        self.length = len(arr)
        self.dummy = dummy

    def __iter__(self):
        def dummy_iterator_function():
            yield self.arr

        def iterator_function():
            for i in range(self.length):
                yield self.doubled_arr[i:i + self.length]

        if self.dummy:
            return dummy_iterator_function()
        else:
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

    arr = [1, 2, 3, 4]
    sg4 = PermutationShiftGenerator(arr, dummy=True)

    print("============ DUMMY ============")

    for i in sg4:
        print(i)


if __name__ == "__main__":
    main()
