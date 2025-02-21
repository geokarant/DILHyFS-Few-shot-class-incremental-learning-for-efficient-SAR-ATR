def get_model(model_name, args):
    name = model_name.lower()
    if name == 'basic':
        from models.basic import Learner
    elif name =='dilhyfs':
        from models.DILHyFS import Learner
    else:
        assert 0
    
    return Learner(args)
