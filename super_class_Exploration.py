class parent: 
    def __init__(self,x=20):
        print("this is the parent class",x) 
        self.val=x
    def func(self):
        print("this is the function of parent class")

class child(parent): 
    def __init__(self,x): 
        super().__init__()
        print("this is the child class",x) 
    def func(self): 
        print(self.val)
        print("this is the function of child class")
    

c = child(20) 
c.func()