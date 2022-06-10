from inspect import Parameter, Signature
fields = ['name', 'shares', 'price']
parms = [ Parameter(name,
        Parameter.POSITIONAL_OR_KEYWORD) for name in fields]
sig = Signature(parms)


def func(*args, **kwargs):
    bound_args = sig.bind(*args, **kwargs)
    for name, val in bound_args.arguments.items():
        print(name, '=', val)


print(sig)
func('ACME', price=91.1, shares=50)