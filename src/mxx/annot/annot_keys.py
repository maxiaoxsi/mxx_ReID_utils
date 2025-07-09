def get_arg_bool(arg):
    if arg == "True":
        return True
    elif arg == "False":
        return False
    raise Exception("[annot_vl] arg value wrong")