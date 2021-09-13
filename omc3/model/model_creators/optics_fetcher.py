class OpticsFecher:
    def get_base_sequence(self):
        raise NotImplementedError("OpticsFetcher needs to implment `get_base_sequence`")
    def get_modifiers(self):
        raise NotImplementedError("OpticsFetcher needs to implment `get_modifiers`")
