# Day_10_04_class.py


class Info:
    def __init__(self, age):
        print(self, 'init')
        self.age = age

    def show(self):
        print('show', self.age)

    def display(self, n):
        print('display', n)


i1 = Info(age=13)
print(i1)

i2 = Info(age=27)
print(i2)

i1.show()
Info.show(i1)
# Info.show('abc')      # error

i1.display(23)
Info.display(i1, 45)

print(i1.age)

# 문제
# i1은 13살, i2는 27살로 만들고 싶어요
# i1.age = 13
# i2.age = 27

i1.show()
i2.show()

