"""Colour tokens for the Qt UI.

Colour constants only -- no QSS strings, no Qt imports -- so this module stays
importable anywhere, including from tooling and tests that do not have the
``[ui]`` extra installed.

These previously lived at the *bottom* of ``fibsem.ui.stylesheets``, below every
QSS constant that would want to consume them, which is why none of those
constants could interpolate a token. Defining them here and importing them at
the top of ``stylesheets`` is what makes the palette the single source of truth
rather than a parallel set of names that happen to agree.

``fibsem.ui.stylesheets`` re-exports every name below, so existing callers doing
``from fibsem.ui.stylesheets import BORDER_COLOR`` keep working unchanged.
"""

# Color palette
PRIMARY_COLOR = "#007ACC"
PRIMARY_COLOR_HOVER = "#118FE4"
PRIMARY_COLOR_PRESSED = "#2d72c4"
GRAY_CANVAS_COLOR = "#000000"
GRAY_CONSOLE_COLOR = "#121212"
GRAY_FOREGROUND_COLOR = "#414851"
GRAY_PRIMARY_COLOR = "#5a626C"
GRAY_HIGHLIGHT_COLOR = "#6A7380"
GRAY_SECONDARY_COLOR = "#868E93"
GRAY_ICON_COLOR = "#D1D2D4"
WHITE_ICON_COLOR = "#ffffff"
GRAY_TEXT_COLOR = "#F0F1F2"
GRAY_WHITE_COLOR = "#FFFFFF"
SEMANTIC_ERROR_COLOR = "#99121F"
SEMANTIC_ERROR_HOVER_COLOR = "#BF2A38"
SEMANTIC_ERROR_PRESSED_COLOR = "#b71c1c"
SEMANTIC_WARNING_COLOR = "#E3B617"
PURPLE_COLOR = "#7C3AED"
ORANGE_COLOR = "#ff9800"
DEFECT_ORANGE_COLOR = "#e8a020"

# Shared canvas colours (image canvas / overlays / quad-view selection border)
CANVAS_BG = "#1e2124"
PRIMARY_ACCENT = "#3a6ea5"  # matches the quad-view selection border

# Stage markers on a real-space canvas. Shared by the fluorescence and FIB/SEM
# overviews on purpose: a user reads both tabs, and yellow meaning "the stage is here"
# on one and something else on the other is worse than either colour alone. They were
# defined privately in the FM overview and would have been copied verbatim into the
# second one, which is how two constants that must agree stop agreeing.
CURRENT_POSITION_COLOUR = "#ffee58"   # where the stage is now
SAVED_POSITION_COLOUR = "#26c6da"     # a marked position
SELECTED_POSITION_COLOUR = "#76ff03"  # the marked position under the selection

# ---------------------------------------------------------------------------
# Napari-dark surface palette
#
# The tokens the hand-built dialogs style themselves from. Named for the role
# they play rather than the shade they are, because that is what a caller is
# choosing: a panel is a panel whether or not it stays #1e2027.
#
# Several of these also existed under shade-derived names. Those names are now
# aliases assigned from the role-named token below (see the bottom of this file)
# rather than second copies of the value, so there is one place to change a
# colour. They are kept, not deleted, because ~90 modules import them.
#
# So: prefer these when styling a new dialog, and do not add a third spelling
# of a colour that is already here twice.
SURFACE_COLOR = "#262930"       # dialog background
PANEL_COLOR = "#1e2027"         # inset panels, table headers
ROW_ALT_COLOR = "#2b2f38"       # alternating row tint, hover
BORDER_COLOR = "#3d4251"        # panel and control borders
TEXT_COLOR = "#d6d6d6"          # body text
TEXT_STRONG_COLOR = "#f0f1f2"   # titles, emphasis
TEXT_MUTED_COLOR = "#868e93"    # secondary text, disabled
ACCENT_COLOR = "#50a6ff"        # links, selected state, informational chips
OK_COLOR = "#4caf50"            # success
WARN_COLOR = "#e0a030"          # loaded but inactive, degraded, needs attention
ERROR_COLOR = "#d04040"         # failure

# The disabled pair. Every semantic button sheet renders its :disabled state in
# these two, and until now both were bare literals repeated across the file --
# the most load-bearing colours in the UI with no name.
#
# DISABLED_BG_COLOR sits 4 units from ROW_ALT_COLOR in RGB, which is to say they
# are the same colour to the eye and different to a diff. Two hover rules in
# NAPARI_STYLE use this value where ROW_ALT_COLOR is the token that means hover;
# collapsing those is a real (if invisible) change, so they are left flagged in
# place rather than folded in silently.
DISABLED_BG_COLOR = "#2d313b"   # disabled control background
DISABLED_TEXT_COLOR = "#6b6b6b" # disabled label and control text

# ---------------------------------------------------------------------------
# Neutral ramp
#
# A second palette, and a deliberate one. Everything above is faintly blue --
# BORDER_COLOR is #3d4251, not a grey -- which is right for the app's chrome and
# wrong behind image data, where a colour cast reads as part of the picture. So
# the plot and canvas widgets built their own neutral scale: 30 pure greys
# across 167 uses, concentrated in minimap_plot_widget and canvas_base.
#
# It was never written down, so it drifted -- #e0e0e0 and #dddddd and #e6e6e6
# all doing the same job. These are the rungs that carry real weight; naming
# them is what stops the next one being invented.
#
# Numbered rather than role-named, unlike the palette above, because a neutral
# does not have one role: #909090 is an axis label here and a disabled glyph
# there. The number tracks darkness, so NEUTRAL_900 is nearly black.
NEUTRAL_200 = "#e0e0e0"
NEUTRAL_300 = "#d0d0d0"
NEUTRAL_400 = "#bbbbbb"
NEUTRAL_450 = "#aaaaaa"
NEUTRAL_500 = "#a0a0a0"
NEUTRAL_550 = "#909090"
NEUTRAL_650 = "#666666"
NEUTRAL_700 = "#606060"
NEUTRAL_750 = "#4a4a4a"
NEUTRAL_800 = "#3a3a3a"
NEUTRAL_850 = "#2a2a2a"
NEUTRAL_900 = "#1a1b1e"

# ---------------------------------------------------------------------------
# Deprecated aliases.
#
# Each of these was a second literal spelling of the colour above it, which is
# how a palette drifts: someone retunes SURFACE_COLOR, GRAY_BACKGROUND_COLOR
# stays behind, and two halves of the app disagree with no diff that shows it.
# Assigned rather than duplicated so that can no longer happen.
#
# Prefer the role-named token in new code. These are kept because ~90 modules
# import them and the rename is not worth the churn in this pass.
GRAY_BACKGROUND_COLOR = SURFACE_COLOR
AUTOMATED_COLOR = OK_COLOR
GREEN_COLOR = OK_COLOR
DEFECT_RED_COLOR = ERROR_COLOR
RED_COLOR = SEMANTIC_ERROR_COLOR

# Not aliased: GRAY_TEXT_COLOR and GRAY_SECONDARY_COLOR hold the same colours as
# TEXT_STRONG_COLOR and TEXT_MUTED_COLOR but spell them in upper case
# ("#F0F1F2" vs "#f0f1f2"). Qt does not care, but the substitution would change
# the rendered string, so it is left for a deliberate decision rather than
# folded into a refactor that is meant to be inert.
