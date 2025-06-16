-- WezTerm Safety Patch
-- Apply this patch to fix critical safety issues in .wezterm.lua

-- CRITICAL SAFETY FIXES for .wezterm.lua
-- Replace the dangerous close actions with safer versions:

-- BEFORE (DANGEROUS):
-- { key = 'w', mods = 'CMD', action = wezterm.action.CloseCurrentTab { confirm = false } },
-- { key = 'Backspace', mods = 'CMD', action = wezterm.action.CloseCurrentPane { confirm = false } },

-- AFTER (SAFE):
local wezterm = require 'wezterm'

-- Safe close actions with confirmation
local safe_close_bindings = {
  -- Safe tab closing with confirmation
  { key = 'w', mods = 'CMD', action = wezterm.action.CloseCurrentTab { confirm = true } },
  
  -- Replace unusual CMD+Backspace with standard CMD+D for pane closing
  { key = 'd', mods = 'CMD', action = wezterm.action.CloseCurrentPane { confirm = true } },
  
  -- Alternative: Keep CMD+Backspace but make it safe
  -- { key = 'Backspace', mods = 'CMD', action = wezterm.action.CloseCurrentPane { confirm = true } },
  
  -- Add alternative force-close for power users (no confirmation)
  { key = 'w', mods = 'CMD|SHIFT', action = wezterm.action.CloseCurrentTab { confirm = false } },
  { key = 'd', mods = 'CMD|SHIFT', action = wezterm.action.CloseCurrentPane { confirm = false } },
}

-- Optional: Environment-based safety mode
local safe_mode = os.getenv("WEZTERM_SAFE_MODE") == "1"
if safe_mode then
  -- In safe mode, ALL close actions require confirmation
  safe_close_bindings = {
    { key = 'w', mods = 'CMD', action = wezterm.action.CloseCurrentTab { confirm = true } },
    { key = 'd', mods = 'CMD', action = wezterm.action.CloseCurrentPane { confirm = true } },
    { key = 'w', mods = 'CMD|SHIFT', action = wezterm.action.CloseCurrentTab { confirm = true } },
    { key = 'd', mods = 'CMD|SHIFT', action = wezterm.action.CloseCurrentPane { confirm = true } },
  }
end

-- Instructions for applying the patch:
--[[
1. Open .wezterm.lua
2. Replace line 99 with:
   { key = 'w', mods = 'CMD', action = wezterm.action.CloseCurrentTab { confirm = true } },

3. Replace line 110 with:
   { key = 'd', mods = 'CMD', action = wezterm.action.CloseCurrentPane { confirm = true } },

4. Optional: Add these additional safe shortcuts after line 110:
   { key = 'w', mods = 'CMD|SHIFT', action = wezterm.action.CloseCurrentTab { confirm = false } },
   { key = 'd', mods = 'CMD|SHIFT', action = wezterm.action.CloseCurrentPane { confirm = false } },

5. Optional: Enable safe mode by setting environment variable:
   export WEZTERM_SAFE_MODE=1
--]]

return safe_close_bindings