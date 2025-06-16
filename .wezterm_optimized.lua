-- Wezterm configuration for Wheel Trading project - PERFORMANCE OPTIMIZED
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
config.font_size = 14.0  -- Increased for better readability during long CLI sessions
config.line_height = 1.1  -- Improved vertical spacing for better readability
config.cell_width = 1.0   -- Standard character width

-- Typography enhancements for M4 Pro display
config.font_antialias = 'Subpixel'  -- Better text rendering on Retina displays
config.font_hinting = 'Full'        -- Improved character sharpness

-- GPU Acceleration for M4 Pro - Optimized for high system load
config.webgpu_preferred_adapter = 'Metal'
config.webgpu_power_preference = 'default'  -- Reduced from 'high-performance'
config.front_end = 'WebGpu'
config.max_fps = 60                         -- Reduced from 120
config.animation_fps = 30                   -- Reduced from 60
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

-- Scrollback and performance for long CLI sessions - Memory optimized
config.scrollback_lines = 5000              -- Reduced from 10K for memory efficiency (Claude Code CLI optimized)
config.enable_scroll_bar = false            -- Disabled to reduce UI memory overhead
config.alternate_screen_scroll_enabled = true  -- Better screen handling

-- Memory optimization settings for M4 Pro 24GB system
config.memory_usage = {
  -- Enable aggressive memory cleanup for long-running sessions
  enable_automatic_gc = true,
  -- Reduce font glyph cache size for better memory efficiency
  max_glyph_cache_size = 1024,  -- MB, reduced from default 2048
}

-- Tab bar - Minimalist for focus
config.hide_tab_bar_if_only_one_tab = true
config.tab_bar_at_bottom = true
config.show_new_tab_button_in_tab_bar = false
config.use_fancy_tab_bar = false

-- Environment variables (defaults, can be overridden by shell) - Memory optimized
local project_root = wezterm.home_dir .. '/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading'
config.set_environment_variables = {
  -- Project identification (using shared variable to reduce memory duplication)
  WHEEL_TRADING_ROOT = project_root,
  
  -- Terminal integration
  WEZTERM_SHELL_INTEGRATION = '1',
  TERM_PROGRAM = 'WezTerm',
  
  -- Claude Code CLI optimization hints
  CLAUDE_CODE_TERMINAL = 'WezTerm',
  CLAUDE_CODE_PROJECT_ROOT = project_root,
  
  -- Memory optimization hints for child processes
  WEZTERM_MEMORY_OPTIMIZED = '1',
  CLAUDE_CODE_MEMORY_LIMIT = '4G',  -- Suggest 4GB limit for Claude Code CLI processes
}

-- Simplified key bindings - Claude Code CLI optimized
config.keys = {
  -- Essential tab management (Mac-standard)
  { key = 't', mods = 'CMD', action = wezterm.action.SpawnTab 'CurrentPaneDomain' },
  { key = 'w', mods = 'CMD', action = wezterm.action.CloseCurrentTab { confirm = false } },
  
  -- Tab navigation (Mac-standard)
  { key = '1', mods = 'CMD', action = wezterm.action.ActivateTab(0) },
  { key = '2', mods = 'CMD', action = wezterm.action.ActivateTab(1) },
  { key = '3', mods = 'CMD', action = wezterm.action.ActivateTab(2) },
  { key = '4', mods = 'CMD', action = wezterm.action.ActivateTab(3) },
  
  -- Pane management (non-conflicting shortcuts)
  { key = 'Enter', mods = 'CMD', action = wezterm.action.SplitHorizontal { domain = 'CurrentPaneDomain' } },
  { key = 'Enter', mods = 'CMD|SHIFT', action = wezterm.action.SplitVertical { domain = 'CurrentPaneDomain' } },
  { key = 'Backspace', mods = 'CMD', action = wezterm.action.CloseCurrentPane { confirm = false } },
  
  -- Simplified pane navigation (CTRL to avoid conflicts)
  { key = 'h', mods = 'CTRL|SHIFT', action = wezterm.action.ActivatePaneDirection 'Left' },
  { key = 'l', mods = 'CTRL|SHIFT', action = wezterm.action.ActivatePaneDirection 'Right' },
  { key = 'k', mods = 'CTRL|SHIFT', action = wezterm.action.ActivatePaneDirection 'Up' },
  { key = 'j', mods = 'CTRL|SHIFT', action = wezterm.action.ActivatePaneDirection 'Down' },
  
  -- Quick launch for essential tools only
  { key = 'l', mods = 'CMD|ALT', action = wezterm.action.SpawnCommandInNewTab {
    label = 'Claude Code CLI',
    args = { 'claude' },
  }},
}

-- PERFORMANCE OPTIMIZED Status bar with memory leak prevention
local last_status_update = 0
local status_cache = {}
local status_cache_timeout = 2000  -- 2 second cache to reduce memory churn
local max_cache_entries = 20       -- OPTIMIZATION: Prevent unbounded cache growth

-- OPTIMIZATION: Cache expensive string operations
local home_dir_pattern = wezterm.home_dir
local home_replacement = "~"

-- OPTIMIZATION: Performance metrics (optional debugging)
local cache_hits = 0
local cache_misses = 0
local update_count = 0

wezterm.on('update-status', function(window, pane)
  local now = wezterm.time.now()
  local pane_id = pane:pane_id()
  
  -- Throttle status updates to reduce memory allocation/deallocation cycles
  if now - last_status_update < status_cache_timeout and status_cache[pane_id] then
    window:set_left_status(status_cache[pane_id])
    cache_hits = cache_hits + 1
    return
  end
  
  cache_misses = cache_misses + 1
  update_count = update_count + 1
  
  -- OPTIMIZATION: Add error handling for pane operations
  local success, cwd = pcall(function()
    local dir = pane:get_current_working_dir()
    return dir and dir.path or nil
  end)
  
  if not success then
    cwd = nil
  end
  
  -- Get user vars from shell integration with error handling
  local success_vars, user_vars = pcall(function()
    return pane:get_user_vars()
  end)
  
  if not success_vars then
    user_vars = {}
  end
  
  local wheel_active = user_vars.wheel_active == "true"
  local claude_active = user_vars.claude_active == "true"
  
  -- OPTIMIZATION: Use table.concat for efficient string building
  local status_parts = {}
  
  -- Claude Code CLI indicator
  if claude_active then
    table.insert(status_parts, '🤖 Claude')
  end
  
  if wheel_active then
    table.insert(status_parts, '🚀 Wheel Trading')
  end
  
  if cwd then
    -- OPTIMIZATION: Use cached pattern for gsub operation
    local short_cwd = cwd:gsub(home_dir_pattern, home_replacement)
    if #short_cwd > 40 then
      short_cwd = "..." .. short_cwd:sub(-37)
    end
    table.insert(status_parts, short_cwd)
  end
  
  -- OPTIMIZATION: Efficient string concatenation
  local formatted_status = #status_parts > 0 and wezterm.format({{Text = table.concat(status_parts, ' | ')}}) or ""
  
  -- OPTIMIZATION: Cache size management to prevent memory leaks
  if next(status_cache) then
    local cache_size = 0
    for _ in pairs(status_cache) do
      cache_size = cache_size + 1
    end
    
    if cache_size >= max_cache_entries then
      -- Remove oldest entry to prevent unbounded growth
      local oldest_pane = nil
      local oldest_time = math.huge
      for cache_pane_id, cache_entry in pairs(status_cache) do
        if type(cache_entry) == "table" and cache_entry.timestamp and cache_entry.timestamp < oldest_time then
          oldest_time = cache_entry.timestamp
          oldest_pane = cache_pane_id
        elseif type(cache_entry) == "string" then
          -- Convert old format to new format with timestamp
          status_cache[cache_pane_id] = {
            status = cache_entry,
            timestamp = now - status_cache_timeout - 1000  -- Mark as old
          }
        end
      end
      if oldest_pane then
        status_cache[oldest_pane] = nil
      end
    end
  end
  
  -- Cache the result with timestamp for cleanup
  status_cache[pane_id] = {
    status = formatted_status,
    timestamp = now
  }
  last_status_update = now
  
  window:set_left_status(formatted_status)
  
  -- OPTIMIZATION: Debug logging (enable only for troubleshooting)
  -- if update_count % 100 == 0 then
  --   wezterm.log_info(string.format("Status updates: %d, Cache hits: %d, Cache misses: %d, Hit ratio: %.1f%%", 
  --     update_count, cache_hits, cache_misses, (cache_hits / (cache_hits + cache_misses)) * 100))
  -- end
end)

-- OPTIMIZATION: Cleanup on pane close to prevent memory leaks
wezterm.on('pane-closed', function(pane, _)
  local pane_id = pane:pane_id()
  if status_cache[pane_id] then
    status_cache[pane_id] = nil
  end
end)

-- Memory-optimized launch menu - Essential commands only (reduced from 17 to 8 items)
config.launch_menu = {
  -- === ESSENTIAL CORE DEVELOPMENT ===
  {
    label = '🤖 Claude Code CLI',
    args = { 'claude' },
    cwd = project_root,
  },
  {
    label = '📁 Project Shell',
    args = { 'zsh' },
    cwd = project_root,
  },
  
  -- === CRITICAL PRODUCTION SYSTEMS ===
  {
    label = '🚀 Einstein+Bolt Complete System',
    args = { 'python', 'start_complete_meta_system.py' },
    cwd = project_root,
  },
  {
    label = '📊 Trading System Diagnose',
    args = { 'python', 'run.py', '--diagnose' },
    cwd = project_root,
  },
  
  -- === ESSENTIAL MONITORING ===
  {
    label = '📈 System Monitor',
    args = { 'python', 'bolt_einstein_resource_monitor.py' },
    cwd = project_root,
  },
  {
    label = '🏥 Health Check',
    args = { 'bash', './production_health_check.sh' },
    cwd = project_root,
  },
  
  -- === ESSENTIAL TESTING ===
  {
    label = '🧪 Fast Tests',
    args = { 'pytest', '-v', '-m', 'not slow' },
    cwd = project_root,
  },
  {
    label = '🚀 M4 Pro Startup',
    args = { 'bash', './startup.sh' },
    cwd = project_root,
  },
}

-- Disable auto-startup behavior to prevent doom scrolling
-- Comment out the gui-startup handler completely
--[[
wezterm.on('gui-startup', function(cmd)
  -- Disabled to prevent automatic command execution and scrolling
end)
--]]

-- Additional memory optimization settings for M4 Pro with 24GB RAM
-- These settings optimize for long-running Claude Code CLI sessions

-- Memory cleanup triggers for long-running sessions
config.exit_behavior = 'Close'  -- Clean exit to prevent memory leaks
config.clean_exit_codes = { 0 }  -- Only clean exit codes to avoid hanging processes

-- Additional performance tweaks for M4 Pro
config.webgpu_preferred_adapter = 'Metal'
config.webgl_preferred_adapter = 'Metal'  -- Fallback GPU acceleration
config.prefer_egl = false  -- Disable EGL for macOS (Metal is better)

-- Font and rendering memory optimizations
config.allow_square_glyphs_to_overflow_width = 'Never'  -- Prevent glyph cache bloat
config.font_rasterizer = 'FreeType'  -- Consistent with other settings
config.font_locator = 'ConfigDirsOnly'  -- Reduce font discovery overhead

-- Memory management for multiple tabs/panes
config.bypass_mouse_reporting_modifiers = 'ALT'  -- Reduce event handling overhead
config.selection_word_boundary = ' \t\n{}[]()"\'-.,;<>'  -- Reduce selection complexity

-- Window management for memory efficiency
config.window_close_confirmation = 'NeverPrompt'  -- Reduce UI overhead
config.adjust_window_size_when_changing_font_size = false  -- Prevent layout thrashing

-- Claude Code CLI specific optimizations
config.automatically_reload_config = false  -- Prevent config reload overhead during sessions
config.check_for_updates = false  -- Disable update checks for stable sessions

-- Tab and pane memory limits (prevent runaway memory usage)
config.max_tabs = 10  -- Reasonable limit for development workflow
config.default_domain = 'local'  -- Keep everything local for performance

-- Memory-efficient key handling (reduced key binding table size)
config.disable_default_key_bindings = false
config.debug_key_events = false  -- Disable debug overhead

return config