"""ETSpy."""

__version__ = "0.8"

from enum import Enum
from typing import List, Literal, Union


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
        return [v.value for k, v in cls.__members__.items()]


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
