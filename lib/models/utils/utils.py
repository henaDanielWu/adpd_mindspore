from collections import OrderedDict
import mindspore
import x2ms_adapter
from x2ms_adapter.auto_static import auto_static

def load_checkpoint(model,
                    filename,
                    map_location='cpu',
                    strict=False,
                    logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    state_dict = get_state_dict(filename, map_location)
    # load state_dict
    x2ms_adapter.load_state_dict(model, state_dict, strict)
    # load_state_dict(model, state_dict, strict, logger)
    return 

def get_state_dict(filename, map_location='cpu'):
    """Get state_dict from a file or URI.

    Args:
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``.
        map_location (str): Same as :func:`torch.load`.

    Returns:
        OrderedDict: The state_dict.
    """
    checkpoint = x2ms_adapter.load(filename, map_location)
    # OrderedDict is a subclass of dict
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    # get state_dict from checkpoint
    if 'state_dict' in checkpoint:
        state_dict_tmp = checkpoint['state_dict']
    else:
        state_dict_tmp = checkpoint

    state_dict = OrderedDict()
    # strip prefix of state_dict
    for k, v in state_dict_tmp.items():
        if k.startswith('module.backbone.'):
            state_dict[k[16:]] = v
        elif k.startswith('module.'):
            state_dict[k[7:]] = v
        elif k.startswith('backbone.'):
            state_dict[k[9:]] = v
        else:
            state_dict[k] = v

    return state_dict
