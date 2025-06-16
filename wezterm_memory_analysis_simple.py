#!/usr/bin/env python3
"""Simple WezTerm Memory Analysis"""

def analyze_wezterm_config():
    """Analyze .wezterm.lua for memory optimizations"""
    
    try:
        with open('.wezterm.lua', 'r') as f:
            config = f.read()
    except FileNotFoundError:
        print("❌ .wezterm.lua not found")
        return
    
    print("🔍 WezTerm Memory Optimization Analysis")
    print("=" * 50)
    
    # Key memory settings to check
    settings = {
        'scrollback_lines': ('5000', 'Scrollback buffer size'),
        'max_glyph_cache_size': ('1024', 'Font glyph cache (MB)'),
        'enable_automatic_gc': ('true', 'Automatic garbage collection'),
        'max_fps': ('60', 'Maximum frame rate'),
        'status_cache_timeout': ('2000', 'Status cache timeout (ms)'),
        'enable_scroll_bar': ('false', 'Scroll bar rendering'),
        'enable_kitty_graphics': ('false', 'Kitty graphics protocol'),
        'macos_window_background_blur': ('0', 'Background blur effects'),
        'automatically_reload_config': ('false', 'Auto config reload'),
        'max_tabs': ('10', 'Maximum tab limit'),
    }
    
    print("\n📊 Memory Settings Analysis:")
    print("-" * 30)
    
    optimized_count = 0
    total_count = len(settings)
    
    for setting, (recommended, description) in settings.items():
        if setting in config:
            # Try to extract value
            lines = [line.strip() for line in config.split('\n') 
                    if setting in line and not line.strip().startswith('--')]
            
            if lines:
                value_found = False
                for line in lines:
                    if '=' in line:
                        try:
                            value = line.split('=')[1].strip().strip(',').strip()
                            if recommended in value:
                                print(f"✅ {setting}: {value} (optimized)")
                                optimized_count += 1
                                value_found = True
                                break
                            else:
                                print(f"⚠️  {setting}: {value} (recommend: {recommended})")
                                value_found = True
                                break
                        except:
                            pass
                
                if not value_found:
                    print(f"📝 {setting}: Found but value unclear")
            else:
                print(f"📝 {setting}: Found in comments only")
        else:
            print(f"❌ {setting}: Not configured")
    
    # Calculate memory estimates
    print("\n💾 Memory Usage Estimates:")
    print("-" * 30)
    
    # Extract scrollback setting
    scrollback_mb = 5.0  # Default estimate
    if 'scrollback_lines = 5000' in config:
        scrollback_mb = 5.0
    elif 'scrollback_lines = 10000' in config:
        scrollback_mb = 10.0
    
    # Extract glyph cache
    glyph_cache_mb = 1024
    if 'max_glyph_cache_size = 1024' in config:
        glyph_cache_mb = 1024
    elif 'max_glyph_cache_size = 2048' in config:
        glyph_cache_mb = 2048
    
    base_overhead = 200  # Base WezTerm overhead
    total_mb = scrollback_mb + glyph_cache_mb + base_overhead
    
    print(f"📊 Estimated per-session usage:")
    print(f"   Scrollback: ~{scrollback_mb} MB")
    print(f"   Glyph cache: {glyph_cache_mb} MB")
    print(f"   Base overhead: ~{base_overhead} MB")
    print(f"   Total: ~{total_mb} MB")
    
    # M4 Pro capacity
    available_ram = 24 * 1024 * 0.7  # 70% of 24GB
    max_sessions = int(available_ram / total_mb)
    
    print(f"\n🖥️  M4 Pro 24GB Capacity:")
    print(f"   Available RAM: ~{available_ram/1024:.1f} GB")
    print(f"   Max sessions: ~{max_sessions}")
    
    # Optimization score
    score_pct = (optimized_count / total_count) * 100
    print(f"\n🎯 Optimization Score: {optimized_count}/{total_count} ({score_pct:.1f}%)")
    
    if score_pct >= 80:
        print("✅ Excellent optimization for long Claude Code sessions!")
    elif score_pct >= 60:
        print("👍 Good optimization")
    else:
        print("⚠️  Needs improvement")
    
    # Check for status bar caching
    print("\n🔄 Status Bar Optimization:")
    print("-" * 30)
    
    if 'status_cache_timeout = 2000' in config:
        print("✅ Status bar caching enabled (2s)")
    else:
        print("❌ Status bar caching not optimized")
    
    if 'status_cache = {}' in config:
        print("✅ Status cache table implemented")
    else:
        print("❌ Status cache table not found")
    
    # Check launch menu
    print("\n📋 Launch Menu Optimization:")
    print("-" * 30)
    
    menu_items = config.count('label =')
    if menu_items <= 10:
        print(f"✅ Launch menu: {menu_items} items (optimized)")
    else:
        print(f"⚠️  Launch menu: {menu_items} items (consider reducing)")
    
    print("\n" + "=" * 50)
    print("✅ Analysis Complete")
    print(f"🎯 Overall: {score_pct:.1f}% optimized for long-running sessions")
    print(f"💾 Memory per session: ~{total_mb:.0f} MB")

if __name__ == "__main__":
    analyze_wezterm_config()