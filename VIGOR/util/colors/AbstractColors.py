class AbstractColors:
    """
    Abstract class for Color subclasses
    """
    def __init__(self):
        self._colors = None

    def __call__(self, color_id):
        raise NotImplementedError("AbstractColors can not be called")

    @property
    def colors(self):
        return self._colors
