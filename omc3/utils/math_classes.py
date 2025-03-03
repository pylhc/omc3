""" 
Math Classes
------------

Provides a mixin class to inherit numerical classes, which ensures 
proper conversion after maths operations.
"""

class MathMixin:
    """ A Mixin Class to be able to inherit from a numerical class and  
        perform operations with it, while returning an instance of the inheriting class. 
        
        Caveats:
         - This assumes that the inheriting class accepts an instance of its parent class 
           in its initialization method !!
         - Use this as the FIRST class in inheritance !
    """

    def __neg__(self):
        result = super().__neg__()
        return self.__class__(result)

    def __add__(self, other):
        result = super().__add__(other)
        return self.__class__(result)

    def __sub__(self, other):
        result = super().__sub__(other)
        return self.__class__(result)

    def __mul__(self, other):
        result = super().__mul__(other)
        return self.__class__(result)

    def __truediv__(self, other):
        result = super().__truediv__(other)
        return self.__class__(result)

    def __pow__(self, other):
        result = super().__pow__(other)
        return self.__class__(result)
    
    # Handle reverse operations 
    def __rtruediv__(self, other):
        result = super().__rtruediv__(other)
        return self.__class__(result)

    def __rpow__(self, other):
        result = super().__rpow__(other)
        return self.__class__(result)

    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__