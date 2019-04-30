from six import iteritems


def write_arguments_to_file(args, filename):
    """
    Save command line argument to log directory

    :param args: argument parser
    :param str filename: file to save argument
    """
    with open(filename, "w") as f:
        for key, value in iteritems(vars(args)):
            f.write("%s: %s\n" % (key, str(value)))