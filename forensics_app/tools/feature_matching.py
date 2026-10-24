"""Set 5.3: match ORB/SIFT keypoints between the current image and a second one."""

from __future__ import annotations

import tkinter as tk

from forensics_app.core import ImageDocument
from .base import ForensicsTool, ToolResult
from .dialogs import TaskCancelled, load_image_file, run_with_progress
from .matching import (
    OUTPUT_ALIGNED,
    OUTPUT_MATCHES,
    OUTPUT_TRANSFORMED,
    MatchSettings,
    apply_feature_matching,
    format_matrix,
    full_resolution_matrix,
    summarize_match,
)
from .matching_dialog import ask_match_settings

OUTPUT_MODES = {
    OUTPUT_MATCHES: "Matches side by side",
    OUTPUT_TRANSFORMED: "Transform the current image into the second one (RANSAC matrix)",
    OUTPUT_ALIGNED: "Transform the second image back into the current one (inverse)",
}


class FeatureMatchingTool(ForensicsTool):
    tool_id = "feature_matching"
    title = "Feature Matching (second image)"
    category = "Feature Detection"
    description = (
        "Matches ORB/SIFT keypoints between the current image and a second image, "
        "with optional RANSAC outlier rejection and alignment."
    )

    def run(self, parent: tk.Misc, document: ImageDocument) -> ToolResult | None:
        assert document.current is not None

        loaded = load_image_file(parent, title="Open the second image to match against")
        if loaded is None:
            return None
        other_path, other = loaded

        answer = ask_match_settings(parent, self.title, MatchSettings(), output_modes=OUTPUT_MODES)
        if answer is None:
            return None
        settings, options = answer
        output = options.get("output", OUTPUT_MATCHES)
        current = document.current

        def task(report, cancelled):
            report(0, 1, "Detecting and matching keypoints…")
            return apply_feature_matching(current, other, settings, output)

        try:
            image, result, features1, features2 = run_with_progress(parent, self.title, task)
        except TaskCancelled:
            return None

        details: dict[str, object] = {
            "Operation": "Feature Matching",
            "Second image": other_path.name,
            "Keypoints (current)": len(features1),
            "Keypoints (second)": len(features2),
            "Matches": result.n_matches,
        }
        if result.inliers is not None:
            details["RANSAC inliers"] = result.n_inliers
            details["Inlier ratio"] = (
                f"{result.n_inliers / result.n_matches:.1%}" if result.n_matches else "n/a"
            )
            if result.model is not None:
                matrix = full_resolution_matrix(result.model, features1.scale, features2.scale)
                details["Transform (current → second)"] = format_matrix(matrix)
        details.update(settings.describe())
        details["Output"] = OUTPUT_MODES[output]

        if output == OUTPUT_TRANSFORMED:
            message = f"Transformed the current image into {other_path.name} ({summarize_match(result)})."
        elif output == OUTPUT_ALIGNED:
            message = f"Transformed {other_path.name} back into the current image ({summarize_match(result)})."
        else:
            message = f"Matched with {other_path.name}: {summarize_match(result)}."
        return ToolResult(message=message, image=image, details=details)
