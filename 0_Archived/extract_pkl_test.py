import library.util as util
import sys


def main(file):
    temp = util.load_pickle(file)
    count = 0
    for i in temp:
        print(i)
        count += 1
    print(count)


if __name__ == '__main__':
    main(sys.argv[1])
