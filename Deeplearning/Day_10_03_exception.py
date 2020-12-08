# Day_10_03_exception.py


try:
    a = [1, 3, 5]
    print(a[0])
    print(a[len(a)])
except IndexError as e:
    print('IndexError')
    print(e)


# 문제
# 아래 코드에서 발생하는 예외를 처리하세요
try:
    b = '3.14'
    print('b :', int(b))
# except ValueError:
#     print('ValueError')
except:
    print('unknown')


while True:
    try:
        number = input('input integer : ')
        number = int(number)
        break
    except ValueError:
        print('정수를 입력하세요')


print(number * 7 % 10)




