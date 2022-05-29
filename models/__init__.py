def create_model(opt):
    if opt['name'] == 'JVSR_base':
        from models.models import JVSRBase as M
    else:
        raise NotImplementedError
    return M(opt)
