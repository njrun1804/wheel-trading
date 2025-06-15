-- Wezterm configuration for Wheel Trading project
local wezterm = require 'wezterm'
local config = {}

-- Use config builder if available
if wezterm.config_builder then
  config = wezterm.config_builder()
end

-- Appearance
config.color_scheme = 'Dracula'
config.font = wezterm.font('JetBrains Mono', { weight = 'Medium' })
config.font_size = 13.0

-- GPU Acceleration for M4 Pro
config.webgpu_preferred_adapter = 'Metal'
config.webgpu_power_preference = 'high-performance'
config.front_end = 'WebGpu'
config.max_fps = 120
config.animation_fps = 60
config.enable_wayland = false
config.enable_kitty_graphics = true

-- Window settings
config.window_decorations = "RESIZE"
config.window_background_opacity = 0.95
config.macos_window_background_blur = 10

-- Tab bar
config.hide_tab_bar_if_only_one_tab = true
config.tab_bar_at_bottom = true
config.show_new_tab_button_in_tab_bar = false
config.use_fancy_tab_bar = false

-- Environment variables (defaults, can be overridden by shell)
config.set_environment_variables = {
  -- These are DEFAULTS - shell can override
  WHEEL_TRADING_ROOT = wezterm.home_dir .. '/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading',
  WEZTERM_SHELL_INTEGRATION = '1',
  TERM_PROGRAM = 'WezTerm',
}

-- Key bindings for split management
config.keys = {
  -- Split panes
  { key = 'd', mods = 'CMD', action = wezterm.action.SplitHorizontal { domain = 'CurrentPaneDomain' } },
  { key = 'd', mods = 'CMD|SHIFT', action = wezterm.action.SplitVertical { domain = 'CurrentPaneDomain' } },
  
  -- Navigate panes
  { key = 'LeftArrow', mods = 'CMD|ALT', action = wezterm.action.ActivatePaneDirection 'Left' },
  { key = 'RightArrow', mods = 'CMD|ALT', action = wezterm.action.ActivatePaneDirection 'Right' },
  { key = 'UpArrow', mods = 'CMD|ALT', action = wezterm.action.ActivatePaneDirection 'Up' },
  { key = 'DownArrow', mods = 'CMD|ALT', action = wezterm.action.ActivatePaneDirection 'Down' },
  
  -- Resize panes
  { key = 'LeftArrow', mods = 'CMD|SHIFT', action = wezterm.action.AdjustPaneSize { 'Left', 5 } },
  { key = 'RightArrow', mods = 'CMD|SHIFT', action = wezterm.action.AdjustPaneSize { 'Right', 5 } },
  
  -- Quick launch configurations
  { key = 'l', mods = 'CMD|SHIFT', action = wezterm.action.SpawnCommandInNewTab {
    label = 'Logs',
    args = { 'bash', '-c', 'scripts/wheel-logs.sh' },
  }},
  { key = 't', mods = 'CMD|SHIFT', action = wezterm.action.SpawnCommandInNewTab {
    label = 'Tests',
    args = { 'bash', '-c', 'scripts/wheel-logs.sh test' },
  }},
  { key = 'j', mods = 'CMD|SHIFT', action = wezterm.action.SpawnCommandInNewTab {
    label = 'Jarvis2',
    args = { 'bash', '-c', 'python -m jarvis2' },
  }},
  { key = 's', mods = 'CMD|SHIFT', action = wezterm.action.SpawnCommandInNewTab {
    label = 'Startup',
    args = { 'bash', '-c', './startup_unified.sh' },
  }},
  { key = 'c', mods = 'CMD|SHIFT', action = wezterm.action.SpawnCommandInNewTab {
    label = 'Claude (Auto-Accept)',
    args = { 'claude', '--dangerously-skip-permissions' },
  }},
}

-- Status bar showing environment info
wezterm.on('update-status', function(window, pane)
  local cwd = pane:get_current_working_dir()
  if cwd then
    cwd = cwd.path
  end
  
  -- Get user vars from shell integration
  local user_vars = pane:get_user_vars()
  local wheel_active = user_vars.wheel_active == "true"
  
  -- Build status based on current state
  local status_items = {}
  
  if wheel_active then
    table.insert(status_items, { Text = '🚀 Wheel Trading' })
  end
  
  if cwd then
    -- Show shortened path
    local short_cwd = cwd:gsub(wezterm.home_dir, "~")
    if #short_cwd > 40 then
      short_cwd = "..." .. short_cwd:sub(-37)
    end
    table.insert(status_items, { Text = ' | ' .. short_cwd })
  end
  
  window:set_left_status(wezterm.format(status_items))
end)

-- Launch menu for quick access to Claude with different modes
config.launch_menu = {
  {
    label = 'Claude (Auto-Accept)',
    args = { 'claude', '--dangerously-skip-permissions' },
  },
  {
    label = 'Claude (Safe Mode)', 
    args = { 'claude' },
  },
  {
    label = 'Trading System',
    args = { 'python', 'run.py' },
    cwd = wezterm.home_dir .. '/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading',
  },
  {
    label = 'Jarvis2',
    args = { 'python', '-m', 'jarvis2' },
    cwd = wezterm.home_dir .. '/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading',
  },
}

-- Disable auto-startup behavior to prevent doom scrolling
-- Comment out the gui-startup handler completely
--[[
wezterm.on('gui-startup', function(cmd)
  -- Disabled to prevent automatic command execution and scrolling
end)
--]]

return config