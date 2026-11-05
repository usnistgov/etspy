"""ETSpy."""

import importlib.metadata

__version__ = importlib.metadata.version("etspy")

import logging
import sys
from collections.abc import Callable
from enum import Enum
from typing import Literal, get_args, get_type_hints


def set_log_level(level: int | str = "INFO"):
    """
    Set the log level of all ETSpy modules and display their messages.

    By default, Python only displays warnings and errors, so ETSpy's
    informational messages (*e.g.* where tilt angles were loaded from, whether
    CUDA was detected, and the shifts and rotations calculated during alignment)
    are hidden. Calling this function sets the level of the ``etspy`` logger and
    adds a handler that prints its messages to standard output. Calling it again
    changes the level without adding another handler. The log levels of other
    packages are not changed.

    Parameters
    ----------
    level
        The log level to set, as accepted by :py:meth:`logging.Logger.setLevel`
        (*e.g.* ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, or ``"ERROR"``). Default
        is ``"INFO"``.

    Examples
    --------
    >>> import etspy.api as etspy
    >>> etspy.set_log_level("INFO")

    Group
    -----
    utilities
    """
    etspy_logger = logging.getLogger("etspy")
    etspy_logger.setLevel(level)
    for handler in etspy_logger.handlers:
        if getattr(handler, "_etspy_handler", False):
            break
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(levelname)s | ETSpy | %(message)s (%(name)s)"),
        )
        handler._etspy_handler = True  # type: ignore[attr-defined]  # noqa: SLF001
        etspy_logger.addHandler(handler)


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
    def values(cls) -> list[str]:
        """Calculate a list of allowed values in the AlignmentMethod enum."""
        return [v.value for _, v in cls.__members__.items()]


AlignmentMethodType = AlignmentMethod | Literal["PC", "COM", "COM-CL", "StackReg"]

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
