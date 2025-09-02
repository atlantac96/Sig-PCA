'''
Taken from https://github.com/jambo6/generalised-signature-method/
'''


from sklearn.base import BaseEstimator, TransformerMixin
import collections as co
import torch
from torch import nn
import signatory

class AddTime(BaseEstimator, TransformerMixin):
    """Add time component to each path.

    For a path of shape [B, L, C] this adds a time channel to be placed at the first index. The time channel will be of
    length L and scaled to exist in [0, 1].
    """
    def fit(self, data, labels=None):
        return self

    def transform(self, data):
        # Batch and length dim
        B, L = data.shape[0], data.shape[1]

        # Time scaled to 0, 1
        time_scaled = torch.linspace(0, 1, L).repeat(B, 1).view(B, L, 1)

        return torch.cat((time_scaled, data), 2)

class AppendZero():
    """ This will append a zero starting vector to every path. """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        zero_vec = torch.zeros(size=(X.size(0), 1, X.size(2)))
        return torch.cat((zero_vec, X), dim=1)

_Pair = co.namedtuple('Pair', ('start', 'end'))
def window_getter(string, **kwargs):
    """Gets the window method correspondent to the given string

    Args:
        string (str): String such that string.title() corresponds to a window method.
        *args: Arguments that will be supplied to the window method.

    Returns:
        Window: An initialised Window method.
    """
    return globals()[string](**kwargs)

class Window:
    """Abstract base class for windows.

    Each subclass should implement __call__, which returns a list of list of 2-tuples. Each 2-tuple specifies the start
    and end of a window. These are then grouped together into a list, and these lists are then grouped together again
    into another list. (Really for the sake of the Dyadic window, which considers windows at multiple scales, so the
    different scales of windows should be grouped together but not grouped with each other.)
    """
    def __init__(self):
        if self.__class__ is Window:
            raise NotImplementedError  # abstract base class

    def num_windows(self, length):
        """ Gets the total number of windows produced by the given window method.

        Args:
            length (int): Length of the time series.

        Returns:
            int: Number of windows.
        """
        all_windows = self(length)
        num_windows = 0
        for window_group in all_windows:
            num_windows += len(window_group)
        return num_windows


class Global(Window):
    """A single window over the whole data."""
    def __call__(self, length=None):
        return [[_Pair(None, None)]]


class _ExpandingSliding(Window):
    def __init__(self, initial_length, start_step, end_step):
        super(_ExpandingSliding, self).__init__()
        if self.__class__ is _ExpandingSliding:
            raise NotImplementedError  # abstract base class
        self.initial_length = initial_length
        self.start_step = start_step
        self.end_step = end_step

    def __call__(self, length):
        def _call():
            start = 0
            end = self.initial_length
            while end <= length:
                yield _Pair(start, end)
                start += self.start_step
                end += self.end_step
        return [list(_call())]


class Sliding(_ExpandingSliding):
    """A window starting at zero and going to some point that increases between windows."""
    def __init__(self, length, step):
        super(Sliding, self).__init__(initial_length=length, start_step=step, end_step=step)


class Expanding(_ExpandingSliding):
    """A window of fixed length, slid along the dataset."""
    def __init__(self, length, step):
        super(Expanding, self).__init__(initial_length=length, start_step=0, end_step=step)


class Dyadic(Window):
    """First the global window over the whole thing. Then the window of the first half and the second. Then the window
    over the first quarter, then the second quarter, then the third quarter, then the fourth quarter, etc. down to
    some specified depth. Make sure the depth isn't too high for the length of the dataset, lest we end up with trivial
    windows that we can't compute a signature over.
    """
    def __init__(self, depth):
        super(Dyadic, self).__init__()
        self.depth = depth

    def __call__(self, length):
        return self.call(float(length))

    def call(self, length, _offset=0.0, _depth=0, _out=None):
        if _out is None:
            _out = [[] for _ in range(self.depth + 1)]
        _out[_depth].append(_Pair(int(_offset), int(_offset + length)))

        if _depth < self.depth:
            left = Dyadic(self.depth)
            right = Dyadic(self.depth)
            half_length = length / 2
            # The order of left then right is important, so that they add their entries to _out in the correct order.
            left.call(half_length, _offset, _depth + 1, _out)
            right.call(half_length, _offset + half_length, _depth + 1, _out)

        return _out

def _compute_length(path):
    differences = path[:, 1:] - path[:, :-1]
    # We use the L^\infty norm on reals^d
    out = differences.abs().max(dim=-1).values
    return out.sum(dim=-1)


def _rescale_path(path, depth, by_length=False):
    # Can approximate this pretty well with Stirling's formula if need be... but we don't need to :P
    coeff = math.factorial(depth) ** (1 / depth)

    if by_length:
        length = _compute_length(path)
        coeff = coeff / length
    return coeff * path


def rescale_path(path, depth):
    """Rescales the input path by depth! ** (1 / depth), so that the last signature term should be roughly O(1)."""
    return _rescale_path(path, depth, False)


def rescale_path_by_length(path, depth):
    """Rescales the input path by depth! ** (1 / depth) / Length(path), so that the last signature term should be
    roughly O(1). (In practice this one seems to do a much worse job of rescaling the path then without the length, so
    it's probably not worth using this one.)
    """
    return _rescale_path(path, depth, True)


def _rescale_signature(signature, channels, depth, length, by_length=False):
    sigtensor_channels = signature.size(-1)
    if signatory.signature_channels(channels, depth) != sigtensor_channels:
        raise ValueError("Given a sigtensor with {} channels, a path with {} channels and a depth of {}, which are "
                         "not consistent.".format(sigtensor_channels, channels, depth))

    if by_length:
        length_reciprocal = length.reciprocal()
        for i in range(len(signature.shape) - len(length.shape)):
            length_reciprocal.unsqueeze_(-1)

    end = 0
    term_length = 1
    val = 1
    terms = []
    for d in range(1, depth + 1):
        start = end
        term_length *= channels
        end = start + term_length

        val *= d
        if by_length:
            val *= length_reciprocal

        terms.append(signature[..., start:end] * val)

    return torch.cat(terms, dim=-1)


def rescale_signature(signature, channels, depth):
    """Rescales the output signature by multiplying the depth-d term by d!, so that every term should be about O(1)."""
    return _rescale_signature(signature, channels, depth, None, False)


def rescale_signature_by_length(signature, path, depth):
    """Rescales the output signature by multiplying the depth-d term by d! * Length(path) ** -d, so that every term
    should be about O(1). In practice this seems to do a worse job normalizing things than without the length."""
    return _rescale_signature(signature, path.size(-1), depth, _compute_length(path), True)

def push_batch_trick(x):
    # If given a 4D tensor, will push the first and second dimensions together to 'hide' the second dimension inside the
    # first (batch) dimension.
    if len(x.shape) == 3:
        return _TrickInfo(False, None, None), x
    elif len(x.shape) == 4:
        return _TrickInfo(True, x.size(0), x.size(1)), x.view(x.size(0) * x.size(1), x.size(2), x.size(3))
    else:
        raise RuntimeError("x has {} dimensions, rather than the expected 3 or 4".format(len(x.shape)))


def unpush_batch_trick(trick_info, x):
    # Once a batch-trick'd tensor has been through the signature then it has lost its stream dimension, so all that's
    # left is a (batch * group, signature_channel)-shaped tensor. This now pushes the trick back the other way, so that
    # the group dimensions becomes part of the channel dimension. (Where it really belongs.)
    if len(x.shape) != 2:
        raise RuntimeError("x has {} dimensions, rather than the expected 2.".format(len(x.shape)))
    if trick_info.tricked:
        return x.view(trick_info.batch_size, trick_info.group_size * x.size(1))
    else:
        return x

ADDITIONAL_PARAM_GRIDS = {
    'window': {
        'Sliding': {
            'small': {
                'num_windows': 5,
            },
            'large': {
                'num_windows': 20
            }
        },
        'Expanding': {
            'small': {
                'num_windows': 5,
            },
            'large': {
                'num_windows': 20
            }
        }
    },

    'dyadic_meta': {
        'small': {
            'hidden_channels': 128,
        },
        'large': {
            'hidden_channels': 512
        }
    }
}
def prepare_window(ds_length, window_name, window_kwargs):
    """Window needs special preparation as the parameters can be dependent on dataset length.

    Args:
        ds_length (int): Length of the dataset.
        window_name (str): Name of the window module.
        window_kwargs (dit): Key word arguments from the grid run.

    Returns:
        window module
    """
    # Variable parameters for ('Expanding'/'Sliding') dependent on the size of the dataset
    if window_name in ['Sliding', 'Expanding']:
        num_windows = ADDITIONAL_PARAM_GRIDS['window'][window_name][window_kwargs['size']]['num_windows']
        length = int(np.floor(ds_length / num_windows))
        window_kwargs = {'length': length, 'step': length}

    window = window_getter(window_name, **window_kwargs)

    return window

class ComputeWindowSignature(nn.Module):
    """ Generic class for computing signatures over windows. """
    def __init__(self,
                 window_name=None,
                 window_kwargs=None,
                 ds_length=None,
                 sig_tfm=None,
                 depth=None,
                 rescaling=None,
                 normalisation=None):
        """
        Args:
            window_name (str): The name of the window transformation to use (must be the name of a class in the window
                module)
            window_kwargs (dict): Dictionary of kwargs to pass to the window class.
            sig_tfm (str): The signature transformation to use 'signature' or 'logsignature'.
            depth (int): The depth to compute the signature up to.
            rescaling (str): Scaling before signature computation or after: 'pre' or 'post'.
            normalisation (
        """
        super(ComputeWindowSignature, self).__init__()
        self.ds_length = ds_length
        self.window_name = window_name
        self.window_kwargs = window_kwargs
        self.ds_length = ds_length
        self.sig_tfm = sig_tfm
        self.depth = depth
        self.normalisation = normalisation

        self.window = prepare_window(ds_length, window_name, window_kwargs)

        # Setup rescaling options
        self.pre_rescaling = lambda path, depth: path
        self.post_rescaling = lambda signature, channels, depth: signature
        if rescaling == 'pre':
            self.pre_rescaling = rescaling_module.rescale_path
        elif rescaling == 'post':
            self.post_rescaling = rescaling_module.rescale_signature

    def _check_inputs(self, window):
        assert isinstance(window, window_module.Window)

    def num_windows(self, length):
        """ Gets the window classes num_windows function. """
        return self.window.num_windows(length)

    def forward(self, path, channels=1, trick_info=False):
        # Rescale
        path = self.pre_rescaling(path, self.depth)

        # Prepare for signature computation
        path_obj = signatory.Path(path, self.depth)
        transform = getattr(path_obj, self.sig_tfm)
        length = path_obj.length

        # Compute signatures in each window returning the grouped list structure
        signatures = []
        for window_group in self.window(length):
            signature_group = []
            for window in window_group:
                signature = transform(window.start, window.end)
                rescaled_signature = self.post_rescaling(signature, path.size(2), self.depth)
                untricked_path = rescaled_signature

                if self.normalisation is not None:
                    untricked_path = self.normalisation(untricked_path)

                if trick_info is not False:
                    untricked_path = unpush_batch_trick(trick_info, rescaled_signature)

                signature_group.append(untricked_path)
            signatures.append(signature_group)

        return signatures

def generate_interaction_terms(channels, depth):
    """
    Generate interaction terms (combinations of channels) for a given depth.
    
    Args:
        channels (int): Number of input features (channels).
        depth (int): The depth of the signature.
        
    Returns:
        terms (list): List of interaction terms.
    """
    feature_indices = list(range(0, channels+1))  # Channels are indexed from 1
    interaction_terms = list(itertools.product(feature_indices, repeat=depth))
    return interaction_terms

def map_column_to_interaction(column_index, channels, max_depth=2):
    """
    Map a column index to its corresponding interaction term.
    
    Args:
        column_index (int): Index of the column in the signature.
        channels (int): Number of input channels/features.
        max_depth (int): Maximum depth to consider (default is 2).
    
    Returns:
        depth (int): The depth level of the interaction.
        interaction_term (tuple): The interaction term for the column index.
    """
    # Compute the number of terms at depth 1
    depth_1_terms = channels + 1  # Includes channels + constant term

    # Check if the column belongs to depth 1
    if column_index < depth_1_terms:
        return 1, (column_index,)  # Feature indices start from 1
    
    # Otherwise, map to depth 2 terms
    # Remove depth 1 columns from index
    column_index -= depth_1_terms

    # Generate depth 2 interaction terms
    depth_2_terms = generate_interaction_terms(channels, 2)
    if column_index-depth_1_terms < len(depth_2_terms):
        return 2, depth_2_terms[column_index]
    
    return None, None  # If index out of range
