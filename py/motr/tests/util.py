def import_motr():
    import os
    import sys
    import inspect
    currdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    parentdir = os.path.dirname(currdir)
    sys.path.insert(0, parentdir)

    import motr

