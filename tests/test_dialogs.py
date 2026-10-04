import unittest
from unittest.mock import MagicMock, patch

from tkinter import simpledialog

from forensics_app.tools import dialogs


class FakeCombobox:
    """Stand-in for ttk.Combobox that remembers the selected index."""

    def __init__(self, *_args, **_kwargs) -> None:
        self._index = -1

    def current(self, index: int | None = None) -> int | None:
        if index is None:
            return self._index
        self._index = index
        return None

    def grid(self, **_kwargs) -> None:
        pass


def unchanged(_dialog) -> None:
    """User action that presses OK without touching the dialog."""


def simulate_dialog(*ok_presses):
    """Replace simpledialog.Dialog.__init__ with its lifecycle, without a window.

    Each argument is one OK press: a function that first edits the open dialog.
    Like the real dialog, a failed validate() keeps it open for the next press,
    and a successful one calls apply(). No presses means the user cancels.
    """

    def fake_init(self, _parent, _title=None) -> None:
        self.result = None
        self.body(MagicMock())
        for edit in ok_presses:
            edit(self)
            if self.validate():
                self.apply()
                return

    return patch.object(simpledialog.Dialog, "__init__", fake_init)


class ChoiceDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ttk = patch.object(dialogs, "ttk").start()
        self.int_var = patch.object(dialogs.tk, "IntVar").start()
        self.addCleanup(patch.stopall)

    def test_creates_one_radio_button_per_option(self) -> None:
        with simulate_dialog(unchanged):
            dialogs.ask_choice(None, "Split", "Channel:", ("Cyan", "Magenta", "Yellow"))

        created = [call.kwargs for call in self.ttk.Radiobutton.call_args_list]
        self.assertEqual([(kw["text"], kw["value"]) for kw in created], [("Cyan", 0), ("Magenta", 1), ("Yellow", 2)])
        self.int_var.assert_called_once()
        self.assertEqual(self.int_var.call_args.kwargs["value"], 0)

    def test_returns_selected_index(self) -> None:
        self.int_var.return_value.get.return_value = 2
        with simulate_dialog(unchanged):
            result = dialogs.ask_choice(None, "Split", "Channel:", ("Cyan", "Magenta", "Yellow"))
        self.assertEqual(result, 2)

    def test_returns_none_when_cancelled(self) -> None:
        with simulate_dialog():
            result = dialogs.ask_choice(None, "Split", "Channel:", ("Cyan", "Magenta"))
        self.assertIsNone(result)


class OrderDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ttk = patch.object(dialogs, "ttk").start()
        self.ttk.Combobox.side_effect = FakeCombobox
        self.warning = patch.object(dialogs.messagebox, "showwarning").start()
        self.addCleanup(patch.stopall)

    def test_returns_initial_order_when_confirmed(self) -> None:
        with simulate_dialog(unchanged):
            result = dialogs.ask_channel_order(None, "Swap", ("Red", "Green", "Blue"), (2, 1, 0))
        self.assertEqual(result, (2, 1, 0))
        self.warning.assert_not_called()

    def test_rejects_repeated_channels(self) -> None:
        with simulate_dialog(unchanged):
            result = dialogs.ask_channel_order(None, "Swap", ("Red", "Green", "Blue"), (0, 0, 1))
        self.assertIsNone(result)
        self.warning.assert_called_once()

    def test_stays_open_until_the_order_is_corrected(self) -> None:
        def pick_blue_for_green(dialog) -> None:
            dialog._selectors[1].current(2)

        with simulate_dialog(unchanged, pick_blue_for_green):
            result = dialogs.ask_channel_order(None, "Swap", ("Red", "Green", "Blue"), (0, 0, 1))
        self.warning.assert_called_once()
        self.assertEqual(result, (0, 2, 1))

    def test_returns_none_when_cancelled(self) -> None:
        with simulate_dialog():
            result = dialogs.ask_channel_order(None, "Swap", ("Red", "Green"), (1, 0))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
