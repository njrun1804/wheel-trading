-- WezTerm GPU acceleration configuration for M4 Pro
-- Add this to your ~/.wezterm.lua or merge with existing config

local wezterm = require 'wezterm'
local config = {}

-- In newer versions of wezterm, use the config_builder
if wezterm.config_builder then
    config = wezterm.config_builder()
end

-- GPU Acceleration Settings for M4 Pro
config.webgpu_preferred_adapter = "Metal"  -- Use Metal on macOS
config.front_end = "WebGpu"                -- Use WebGPU renderer
config.max_fps = 120                       -- M4 Pro can handle 120fps easily
config.animation_fps = 60                  -- Smooth animations
config.enable_wayland = false              -- Not needed on macOS

-- Performance optimizations
config.scrollback_lines = 10000            -- Reasonable scrollback
config.enable_scroll_bar = false           -- Reduce rendering overhead
config.use_fancy_tab_bar = true            -- GPU accelerated tab bar

-- Font rendering optimizations
config.freetype_load_target = "Light"      -- Better font rendering
config.freetype_render_target = "HorizontalLcd"

-- Color scheme optimized for readability
config.color_scheme = 'Dracula'            -- Or your preferred scheme

-- Window settings
config.window_background_opacity = 0.95    -- Slight transparency (GPU composited)
config.macos_window_background_blur = 20   -- macOS blur effect

-- Key bindings for Jarvis2 quick access
config.keys = {
    -- Cmd+Shift+J to launch Jarvis2 in new tab
    {
        key = 'j',
        mods = 'CMD|SHIFT',
        action = wezterm.action.SpawnCommandInNewTab {
            args = { 'python', '-m', 'jarvis2', '--interactive' },
            cwd = wezterm.home_dir .. '/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading',
        },
    },
    -- Cmd+Shift+T to launch trading system
    {
        key = 't',
        mods = 'CMD|SHIFT',
        action = wezterm.action.SpawnCommandInNewTab {
            args = { 'python', 'run.py' },
            cwd = wezterm.home_dir .. '/Library/Mobile Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading',
        },
    },
}

-- Tab bar settings
config.tab_bar_at_bottom = false
config.use_fancy_tab_bar = true
config.tab_max_width = 32

-- Startup program (run the unified startup script)
config.default_prog = {
    '/bin/bash',
    '-l',
    '-c',
    'cd ~/Library/Mobile\\ Documents/com~apple~CloudDocs/pMike/Wheel/wheel-trading && ./startup_unified.sh; exec $SHELL'
}

-- Environment variables for hardware acceleration
config.set_environment_variables = {
    -- Jarvis2 optimization
    JARVIS2_BACKEND_PREFERENCE = "mlx,mps,cpu",
    JARVIS2_MEMORY_LIMIT_GB = "18",
    
    -- Metal/GPU settings
    PYTORCH_ENABLE_MPS_FALLBACK = "1",
    PYTORCH_METAL_WORKSPACE_LIMIT_BYTES = tostring(18 * 1024 * 1024 * 1024),
    
    -- CPU optimization
    OMP_NUM_THREADS = "12",
    
    -- WezTerm GPU
    WEZTERM_ENABLE_WEBGPU = "1",
}

return config