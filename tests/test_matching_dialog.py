"""Tests for the matching settings form, driven without a display."""

import unittest
from unittest.mock import MagicMock, patch

from tkinter import simpledialog

from forensics_app.tools import matching_dialog
from forensics_app.tools.matching import MatchSettings


class FakeVar:
    """Stand-in for tk.StringVar / tk.BooleanVar."""

    def __init__(self, _master=None, value=None) -> None:
        self.value = value

    def get(self):
        return self.value

    def set(self, value) -> None:
        self.value = value


class FakeCombobox:
    """Stand-in for ttk.Combobox that remembers the selected index and its state."""

    def __init__(self, *_args, **kwargs) -> None:
        self.values = kwargs.get("values", [])
        self.index = -1
        self.state = None

    def current(self, index: int | None = None) -> int | None:
        if index is None:
            return self.index
        self.index = index
        return None

    def configure(self, **kwargs) -> None:
        self.state = kwargs.get("state", self.state)

    def grid(self, **_kwargs) -> None:
        pass


class FakeEntry:
    def __init__(self, *_args, **_kwargs) -> None:
        self.state = None

    def configure(self, **kwargs) -> None:
        self.state = kwargs.get("state", self.state)

    def grid(self, **_kwargs) -> None:
        pass


def unchanged(_dialog) -> None:
    """User action that presses OK without touching the form."""


def simulate_dialog(*ok_presses):
    """Run the dialog lifecycle without a window: each argument edits the form, then OK.

    A failed validate() keeps the dialog open for the next press; no presses = Cancel.
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


def set_field(name: str, value):
    def edit(dialog) -> None:
        dialog._vars[name].set(value)

    return edit


class MatchSettingsDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        ttk = patch.object(matching_dialog, "ttk").start()
        ttk.Entry.side_effect = FakeEntry
        ttk.Combobox.side_effect = FakeCombobox
        patch.object(matching_dialog.tk, "StringVar", FakeVar).start()
        patch.object(matching_dialog.tk, "BooleanVar", FakeVar).start()
        self.warning = patch.object(matching_dialog.messagebox, "showwarning").start()
        self.addCleanup(patch.stopall)

    def ask(self, *presses, initial=MatchSettings(), **kwargs):
        with simulate_dialog(*presses):
            return matching_dialog.ask_match_settings(None, "Match", initial, **kwargs)

    def test_ok_without_changes_returns_the_initial_settings(self) -> None:
        settings, options = self.ask(unchanged)
        self.assertEqual(settings, MatchSettings())
        self.assertEqual(options, {})

    def test_cancel_returns_none(self) -> None:
        self.assertIsNone(self.ask())

    def test_edited_fields_are_parsed(self) -> None:
        def edit(dialog) -> None:
            dialog._method.set("orb")
            dialog._vars["n_keypoints"].set("800")
            dialog._vars["max_ratio"].set("0.9")
            dialog._vars["cross_check"].set(False)
            dialog._widgets["ransac_model"].current(1)

        settings, _ = self.ask(edit)
        self.assertEqual(settings.method, "orb")
        self.assertEqual(settings.n_keypoints, 800)
        self.assertAlmostEqual(settings.max_ratio, 0.9)
        self.assertFalse(settings.cross_check)
        self.assertEqual(settings.ransac_model, "projective")

    def test_invalid_value_warns_and_keeps_the_dialog_open(self) -> None:
        settings, _ = self.ask(set_field("n_keypoints", "lots"), set_field("n_keypoints", "300"))
        self.warning.assert_called_once()
        self.assertIn("whole number", self.warning.call_args.args[1])
        self.assertEqual(settings.n_keypoints, 300)

    def test_invalid_value_and_cancel_returns_none(self) -> None:
        self.assertIsNone(self.ask(set_field("max_ratio", "3")))
        self.warning.assert_called_once()

    def test_output_and_recursive_options_are_returned(self) -> None:
        modes = {"matches": "Matches", "transformed": "Transform"}

        def edit(dialog) -> None:
            dialog._output.set("transformed")
            dialog._recursive.set(True)

        _, options = self.ask(edit, output_modes=modes, ask_recursive=True)
        self.assertEqual(options, {"output": "transformed", "recursive": True})

    def test_first_output_mode_is_the_default(self) -> None:
        _, options = self.ask(unchanged, output_modes={"matches": "Matches", "aligned": "Aligned"})
        self.assertEqual(options["output"], "matches")

    def test_warp_outputs_need_ransac(self) -> None:
        def edit(dialog) -> None:
            dialog._output.set("aligned")
            dialog._vars["use_ransac"].set(False)

        result = self.ask(edit, output_modes={"matches": "Matches", "aligned": "Aligned"})
        self.assertIsNone(result)
        self.assertIn("needs RANSAC", self.warning.call_args.args[1])

    def test_fields_are_enabled_only_when_they_apply(self) -> None:
        states = {}

        def edit(dialog) -> None:
            dialog._update_states()
            states["sift"] = {name: widget.state for name, widget in dialog._widgets.items()}
            dialog._method.set("orb")
            dialog._vars["use_ransac"].set(False)
            dialog._update_states()
            states["orb"] = {name: widget.state for name, widget in dialog._widgets.items()}

        self.ask(edit)
        self.assertEqual(states["sift"]["n_keypoints"], "disabled")
        self.assertEqual(states["sift"]["n_octaves"], "normal")
        self.assertEqual(states["sift"]["ransac_model"], "readonly")
        self.assertEqual(states["orb"]["n_keypoints"], "normal")
        self.assertEqual(states["orb"]["sigma"], "disabled")
        self.assertEqual(states["orb"]["residual_threshold"], "disabled")
        self.assertEqual(states["orb"]["ransac_model"], "disabled")

    def test_initial_values_fill_the_form(self) -> None:
        initial = MatchSettings(method="orb", n_keypoints=123, ransac_model="projective")
        captured = {}

        def edit(dialog) -> None:
            captured["n_keypoints"] = dialog._vars["n_keypoints"].get()
            captured["model"] = dialog._widgets["ransac_model"].current()

        settings, _ = self.ask(edit, initial=initial)
        self.assertEqual(captured, {"n_keypoints": "123", "model": 1})
        self.assertEqual(settings, initial)


if __name__ == "__main__":
    unittest.main()
