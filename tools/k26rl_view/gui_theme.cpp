/* gui_theme.cpp - the Graphite palette applied to the
 * interface, and nothing of the scene. */
#include "gui_internal.h"

namespace k26rl_view {

/* The palette.
 *
 * Grey chrome with one amber accent, and a red reserved for a
 * failure. Hue carries meaning only where something is an accent or
 * an error, so nothing else in the interface introduces one: a
 * reading is told apart from its neighbours by position and weight
 * rather than by colour, which is what lets a panel drop the sentence
 * explaining what it is.
 *
 * The field wells are dark, and deliberately. Dear ImGui takes one
 * text colour per frame, so a light well under this text colour
 * renders at a contrast ratio near one and the field is unreadable
 * without pushing a colour around every widget in the tree.
 *
 * The values are display-encoded bytes over 255, straight alpha.
 */

ImVec4 rgb_(unsigned hex, float a)
{
    return ImVec4((float)((hex >> 16) & 0xFFu) / 255.0f,
                  (float)((hex >> 8) & 0xFFu) / 255.0f,
                  (float)(hex & 0xFFu) / 255.0f, a);
}

/* Every slot is written before the named ones, so a colour this build
 * does not name can never bleed a default through. */

void apply_theme_(ImGuiStyle &st)
{
    for (int i = 0; i < ImGuiCol_COUNT; i++)
        st.Colors[i] = rgb_(COL_TEXT);
    st.Colors[ImGuiCol_Text]                  = rgb_(COL_TEXT);
    st.Colors[ImGuiCol_TextDisabled]          = rgb_(COL_TEXT_DIM);
    st.Colors[ImGuiCol_WindowBg]              = rgb_(COL_WINDOW_BG);
    st.Colors[ImGuiCol_ChildBg]               = rgb_(COL_WINDOW_BG, 0.0f);
    st.Colors[ImGuiCol_PopupBg]               = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_Border]                = rgb_(COL_BORDER);
    st.Colors[ImGuiCol_BorderShadow]          = rgb_(COL_WINDOW_BG, 0.0f);
    st.Colors[ImGuiCol_FrameBg]               = rgb_(COL_INPUT_BG);
    st.Colors[ImGuiCol_FrameBgHovered]        = rgb_(COL_INPUT_BG, 0.85f);
    st.Colors[ImGuiCol_FrameBgActive]         = rgb_(COL_ACCENT_SECOND, 0.55f);
    st.Colors[ImGuiCol_TitleBg]               = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TitleBgActive]         = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TitleBgCollapsed]      = rgb_(COL_SURFACE_BG, 0.85f);
    st.Colors[ImGuiCol_MenuBarBg]             = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_ScrollbarBg]           = rgb_(COL_WINDOW_BG);
    st.Colors[ImGuiCol_ScrollbarGrab]         = rgb_(COL_BORDER);
    st.Colors[ImGuiCol_ScrollbarGrabHovered]  = rgb_(COL_TEXT_DIM);
    st.Colors[ImGuiCol_ScrollbarGrabActive]   = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_CheckMark]             = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_SliderGrab]            = rgb_(COL_TEXT_DIM);
    st.Colors[ImGuiCol_SliderGrabActive]      = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_Button]                = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_ButtonHovered]         = rgb_(COL_HOVER);
    st.Colors[ImGuiCol_ButtonActive]          = rgb_(COL_ACCENT_SECOND, 0.55f);
    st.Colors[ImGuiCol_Header]                = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_HeaderHovered]         = rgb_(COL_HOVER);
    st.Colors[ImGuiCol_HeaderActive]          = rgb_(COL_BORDER, 0.85f);
    st.Colors[ImGuiCol_Separator]             = rgb_(COL_BORDER);
    st.Colors[ImGuiCol_SeparatorHovered]      = rgb_(COL_TEXT_DIM);
    st.Colors[ImGuiCol_SeparatorActive]       = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_ResizeGrip]            = rgb_(COL_BORDER, 0.35f);
    st.Colors[ImGuiCol_ResizeGripHovered]     = rgb_(COL_TEXT_DIM, 0.55f);
    st.Colors[ImGuiCol_ResizeGripActive]      = rgb_(COL_ACCENT, 0.55f);
    st.Colors[ImGuiCol_Tab]                   = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TabHovered]            = rgb_(COL_HOVER);
    st.Colors[ImGuiCol_TabActive]             = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TabUnfocused]          = rgb_(COL_WINDOW_BG);
    st.Colors[ImGuiCol_TabUnfocusedActive]    = rgb_(COL_SURFACE_BG, 0.85f);
    /* The docking chrome. The central node is left transparent
     * because the scene is drawn behind it, so the node's own
     * background must not paint over the picture. */
    st.Colors[ImGuiCol_DockingPreview]        = rgb_(COL_ACCENT, 0.35f);
    st.Colors[ImGuiCol_DockingEmptyBg]        = rgb_(COL_WINDOW_BG, 0.0f);
    st.Colors[ImGuiCol_PlotLines]             = rgb_(COL_TEXT);
    st.Colors[ImGuiCol_PlotLinesHovered]      = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_PlotHistogram]         = rgb_(COL_TEXT);
    st.Colors[ImGuiCol_PlotHistogramHovered]  = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_TableHeaderBg]         = rgb_(COL_SURFACE_BG);
    st.Colors[ImGuiCol_TableBorderStrong]     = rgb_(COL_BORDER);
    st.Colors[ImGuiCol_TableBorderLight]      = rgb_(COL_BORDER, 0.55f);
    st.Colors[ImGuiCol_TableRowBg]            = rgb_(COL_WINDOW_BG, 0.0f);
    st.Colors[ImGuiCol_TableRowBgAlt]         = rgb_(COL_SURFACE_BG, 0.35f);
    st.Colors[ImGuiCol_TextSelectedBg]        = rgb_(COL_ACCENT_SECOND, 0.35f);
    st.Colors[ImGuiCol_DragDropTarget]        = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_NavHighlight]          = rgb_(COL_ACCENT);
    st.Colors[ImGuiCol_NavWindowingHighlight] = rgb_(COL_ACCENT, 0.55f);
    st.Colors[ImGuiCol_NavWindowingDimBg]     = rgb_(COL_WINDOW_BG, 0.55f);
    st.Colors[ImGuiCol_ModalWindowDimBg]      = rgb_(COL_WINDOW_BG, 0.55f);
}

}  /* namespace k26rl_view */
