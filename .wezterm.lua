-- Wezterm configuration for Wheel Trading project
local wezterm = require 'wezterm'
local config = {}

-- Use config builder if available
if wezterm.config_builder then
  config = wezterm.config_builder()
end

-- Appearance - Optimized for Claude Code CLI intensive work
config.color_scheme = 'Dracula'
config.font = wezterm.font('JetBrains Mono', { 
  weight = 'Medium',
  stretch = 'Normal',
  style = 'Normal'
})
-- Font size set below for Claude Code CLI
config.cell_width = 1.0   -- Standard character width

-- Typography enhancements for M4 Pro display
config.font_antialias = 'Subpixel'  -- Better text rendering on Retina displays
config.font_hinting = 'Full'        -- Improved character sharpness

-- GPU Acceleration for M4 Pro - Optimized for high system load
config.webgpu_preferred_adapter = 'Metal'
config.webgpu_power_preference = 'default'  -- Reduced from 'high-performance'
config.front_end = 'WebGpu'
-- FPS settings moved to Claude CLI section below
config.enable_wayland = false
config.enable_kitty_graphics = false       -- Disabled to reduce GPU load

-- Window settings - Optimized for Claude Code CLI work
config.window_decorations = "RESIZE"
config.window_background_opacity = 1.0     -- Solid background for better text contrast
config.macos_window_background_blur = 0    -- Disabled blur effects for performance
config.window_padding = {                  -- Comfortable padding for long CLI sessions
  left = 8,
  right = 8,
  top = 8,
  bottom = 8,
}

-- Display optimization for M4 Pro MacBook
config.dpi = nil                           -- Auto-detect DPI for proper scaling
config.freetype_load_target = 'Normal'     -- Balanced rendering
config.freetype_render_target = 'Normal'   -- Consistent with load target

-- Cursor and editing optimizations for CLI work
config.default_cursor_style = 'SteadyBlock'  -- Clear cursor visibility
config.cursor_blink_rate = 800              -- Comfortable blink rate
config.cursor_thickness = '2px'             -- Slightly thicker for visibility

-- Scrollback settings moved to Claude CLI section below
config.alternate_screen_scroll_enabled = true  -- Better screen handling

-- Simplified configuration without custom memory management

-- Tab bar - Minimalist for focus
config.hide_tab_bar_if_only_one_tab = true
config.tab_bar_at_bottom = true
config.show_new_tab_button_in_tab_bar = false
config.use_fancy_tab_bar = false

-- Environment variables moved to Claude CLI section below

-- Claude Code CLI optimized key bindings
local act = wezterm.action
config.keys = {
  -- Silence CMD-m to avoid conflicts with Claude's /m and /mcp commands
  {key='m', mods='CMD', action=act.DisableDefaultAssignment},
  
  -- Removed split pane binding to prevent automatic splits
  
  -- Clear scrollback for fresh Claude runs
  {key='K', mods='CTRL|SHIFT', action=act.ClearScrollback('ScrollbackAndViewport')},
  
  -- Quick "yes" for Claude permission prompts
  {key='Y', mods='CMD', action=act.SendString("yes\n")},
}

-- Disable custom status bar
-- wezterm.on('update-status', function(window, pane)
-- end)

-- Launch menu disabled to prevent automatic splits
-- config.launch_menu = {
--   { label = "Claude Code", args = {"claude"} },
--   { label = "Project Shell", args = {"zsh"} },
-- }

-- Disable auto-startup behavior to prevent doom scrolling
-- Comment out the gui-startup handler completely
--[[
wezterm.on('gui-startup', function(cmd)
  -- Disabled to prevent automatic command execution and scrolling
end)
--]]

-- Claude Code CLI optimizations
config.exit_behavior = 'Close'
config.scrollback_lines = 20000  -- Extended for Claude's long outputs
config.enable_scroll_bar = true   -- Visual navigation for long responses

-- Silence terminal bells during Claude streaming
config.audible_bell = "Disabled"

-- Performance for Claude's heavy text output
config.max_fps = 60
config.animation_fps = 30

-- Better for reading Claude's formatted output
config.line_height = 1.2
config.font_size = 13.0

-- Environment variables for Claude Code CLI
config.set_environment_variables = {
  TERM_PROGRAM = 'WezTerm',
  CLAUDE_CODE_TERMINAL = '1',
  CLAUDE_MODEL = 'sonnet',  -- Default model
  NODE_OPTIONS = '--max-old-space-size=20480 --max-semi-space-size=1024',  -- Memory for large outputs
}

return config