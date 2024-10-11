"""ETSpy."""
import importlib.metadata

__version__ = importlib.metadata.version("etspy")

from enum import Enum
from typing import Callable, List, Literal, Union, get_args, get_type_hints


class AlignmentMethod(str, Enum):
    """
    Allowed values for the stack alignment method.

    See :py:func:`etspy.align.align_stack` for more details.

    Group
    -----
    align
    """

    STACK_REG = "StackReg"
    "Stack Registration method"

    PC = "PC"
    "Phase correlation method"

    COM = "COM"
    "Center of Mass method"

    COM_CL = "COM-CL"
    "Center of Mass with Common Line method"

    @classmethod
    def is_valid_value(cls, value) -> bool:
        """Test if value is contained in the AlignmentMethod enum."""
        try:
            cls(value)
        except ValueError:
            return False
        else:
            return True

    @classmethod
    def values(cls) -> List[str]:
        """Calculate a list of allowed values in the AlignmentMethod enum."""
        return [v.value for _, v in cls.__members__.items()]


AlignmentMethodType = Union[
    AlignmentMethod,
    Literal["PC", "COM", "COM-CL", "StackReg"],
]

FbpMethodType = Literal[
    "ram-lak",
    "shepp-logan",
    "cosine",
    "hamming",
    "hann",
    "none",
    "tukey",
    "lanczos",
    "triangular",
    "gaussian",
    "barlett-hann",
    "blackman",
    "nuttall",
    "blackman-harris",
    "blackman-nuttall",
    "flat-top",
    "kaiser",
    "parzen",
    "projection",
    "sinogram",
    "rprojection",
    "rsinogram",
]

ReconMethodType = Literal["FBP", "SIRT", "SART", "DART"]


def _get_literal_hint_values(function: Callable, param_name: str) -> tuple:
    """Get values specified by a Literal type for a given function and parameter."""
    return get_args(get_type_hints(function)[param_name])


def _format_choices(choices: list | tuple) -> str:
    """
    Format a list of values as a string showing options.

    For example, the tuple ("one", "two", "three") would be
    formatted as '["one", "two", or "three"]'. This method is helpful
    for printing context in error messages.
    """
    first_part = ", ".join(
        [f'"{i}"' if isinstance(i, str) else str(i) for i in choices[:-1]],
    )
    middle_part = ", " if len(choices[:-1]) > 1 else " "
    last_part = "or " + (
        f'"{choices[-1]}"' if isinstance(choices[-1], str) else str(choices[-1])
    )
    return f"[{first_part}{middle_part}{last_part}]"
