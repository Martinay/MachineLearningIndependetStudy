from multiprocessing import Process

def test():
    print('ausgabeP')

p2 = Process(target=test)

print('ausgabe')
p2.start()
print('ausgabe')
p2.join()