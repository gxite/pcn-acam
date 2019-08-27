import library.util as util

temp = util.load_pickle('./test.pkl')
count = 0
for i in temp:
    print(i)
    count += 1
print(count)