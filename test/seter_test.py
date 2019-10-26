class A():
    def __init__(self):
        self.name="jingenyan"
        self.language="eng"
    @property
    def _name(self):
        return self.name

    @_name.setter
    def _name(self,value):
        self.name=value

a=A()
a._name="caiyun"
print(a._name)
a.__