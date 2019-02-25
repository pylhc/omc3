"""
Module parser.entry_datatypes
-----------------------------

Advanced Datatypes to add as type to entrypoint. Or any parser, really.

"""
import abc

TRUE_ITEMS = ["True", "1", True, 1]  # items that count as True
FALSE_ITEMS = ["False", "0", False, 0]  # items that count as False


# Meta Class Helper ############################################################


def get_instance_faker_meta(*classes):
    """ Returns the metaclass that fakes the isinstance() and issubclass() checks. """
    class FakeMeta(abc.ABCMeta):
        def __instancecheck__(cls, inst):
            return isinstance(inst, classes)

        def __subclasscheck__(self, subclass):
            return any([issubclass(c, subclass) for c in classes])

    return FakeMeta


# 'Merge' Classes ##############################################################


def get_multi_class(*classes):
    """ Create a class 'behaving' like all classes in `classes`.

    In case a value needs to be converted to a class in this list,
    it is attempted to cast the input to the classes in the given order
    (i.e. string-classes need to go to the end, as they 'always' succeed).
    """
    class MultiClass(metaclass=get_instance_faker_meta(*classes)):

        @classmethod
        def _convert_to_a_type(cls, value):
            for c in classes:
                try:
                    return c.__new__(c, value)
                except (ValueError, TypeError):
                    pass
            else:
                cls_string = ','.join([c.__name__ for c in classes])
                raise ValueError(
                    f"The value '{value}' cant be converted to any of the classes '{cls_string:s}'"
                )

        def __new__(cls, value):
            if isinstance(value, str) or not isinstance(value, classes):
                return cls._convert_to_a_type(value)
            return value

    return MultiClass


# More Fake Classes ############################################################


class DictAsString(metaclass=get_instance_faker_meta(str, dict)):
    """ Use dicts in command line like {"key":value} """
    def __new__(cls, s):
        if isinstance(s, dict):
            return s

        d = eval(s)
        if not isinstance(d, dict):
            raise ValueError(f"'{s}' can't be converted to a dictionary.")
        return d


class BoolOrString(metaclass=get_instance_faker_meta(bool, str)):
    """ A class that behaves like a boolean when possible, otherwise like a string."""
    def __new__(cls, value):
        if isinstance(value, str):
            value = value.strip("\'\"")  # behavior like dict-parser

        if value in TRUE_ITEMS:
            return True

        elif value in FALSE_ITEMS:
            return False

        else:
            return str(value)


class BoolOrList(metaclass=get_instance_faker_meta(bool, list)):
    """ A class that behaves like a boolean when possible, otherwise like a list.

    Hint: 'list.__new__(list, value)' returns an empty list."""
    def __new__(cls, value):
        if value in TRUE_ITEMS:
            return True

        elif value in FALSE_ITEMS:
            return False
        else:
            if isinstance(value, str):
                value = eval(value)
            return list(value)
