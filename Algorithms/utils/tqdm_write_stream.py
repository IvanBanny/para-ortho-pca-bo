"""Stream utility for redirecting stdout to tqdm.write without infinite recursion.

This module provides a custom stream class that allows safely redirecting
standard output to tqdm's write method without causing recursion errors.
"""

import sys
from typing import Optional, Any
from tqdm import tqdm


class TqdmWriteStream:
    """A stream wrapper that redirects stdout to tqdm.write without causing recursion.

    This class solves the infinite recursion problem that occurs when redirecting
    sys.stdout to tqdm.write directly, by temporarily restoring the original stdout
    during the write operation.

    Attributes:
        pbar (tqdm): The tqdm progress bar to write output to.
        _original_stdout: The original stdout stream that was replaced.
    """

    def __init__(self, pbar: tqdm, original_stdout: Optional[Any] = None):
        """Initialize the TqdmWriteStream.

        Args:
            pbar (tqdm): The tqdm progress bar to write output to.
            original_stdout (Optional): The original stdout stream. If None,
                the current sys.stdout will be used.
        """
        self.pbar = pbar
        self._original_stdout = original_stdout if original_stdout is not None else sys.stdout

    def write(self, s: str) -> None:
        """Write the given string to the progress bar.

        This method temporarily restores the original stdout before calling
        pbar.write() to prevent recursion loops.

        Args:
            s (str): The string to write.
        """
        # Only process non-empty strings (skip whitespace-only strings)
        if s.strip():
            # Temporarily restore original stdout to prevent recursion
            current_stdout = sys.stdout
            sys.stdout = self._original_stdout

            try:
                self.pbar.write(s)
            finally:
                # Restore our custom stdout
                sys.stdout = current_stdout

    def flush(self) -> None:
        """Implement flush method (required for stream-like objects)."""
        pass


def redirect_stdout_to_tqdm(pbar: tqdm) -> sys.stdout:
    """Redirect sys.stdout to write to a tqdm progress bar.

    Args:
        pbar (tqdm): The tqdm progress bar to redirect output to.

    Returns:
        The original sys.stdout object, which should be restored when done.
    """
    original_stdout = sys.stdout
    sys.stdout = TqdmWriteStream(pbar, original_stdout)
    return original_stdout


def restore_stdout(original_stdout: Any) -> None:
    """Restore the original stdout after redirection.

    Args:
        original_stdout: The original stdout to restore.
    """
    sys.stdout = original_stdout
