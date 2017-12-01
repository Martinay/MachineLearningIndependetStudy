class test:
    abc = 'hallo'
    def newAttrib(self):
        setattr(self, 'newa', 4)
        setattr(self, 'abc', 2)

a = test()
print(a.abc)
a.newAttrib()
print(a.abc)
print(a.newa)
