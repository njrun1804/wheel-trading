#!/usr/bin/env node

/**
 * M4 Pro Node.js Memory Monitor
 * Prevents string overflow by monitoring memory pressure
 */

const os = require('os');
const v8 = require('v8');

class MemoryMonitor {
    constructor(options = {}) {
        this.maxHeapWarning = options.maxHeapWarning || 0.85; // 85% of heap limit
        this.maxStringLength = options.maxStringLength || 500 * 1024 * 1024; // 500MB strings
        this.checkInterval = options.checkInterval || 5000; // 5 seconds
        this.alertCallback = options.alertCallback || this.defaultAlert;
        
        this.monitoring = false;
        this.stats = {
            gcCount: 0,
            lastGCTime: Date.now(),
            maxHeapUsed: 0,
            pressureEvents: 0
        };
    }

    getMemoryStats() {
        const memUsage = process.memoryUsage();
        const heapStats = v8.getHeapStatistics();
        const systemMem = {
            total: os.totalmem(),
            free: os.freemem(),
            used: os.totalmem() - os.freemem()
        };

        return {
            process: {
                rss: memUsage.rss,
                heapTotal: memUsage.heapTotal,
                heapUsed: memUsage.heapUsed,
                external: memUsage.external,
                arrayBuffers: memUsage.arrayBuffers
            },
            heap: {
                totalSize: heapStats.total_heap_size,
                usedSize: heapStats.used_heap_size,
                limit: heapStats.heap_size_limit,
                usagePercent: (heapStats.used_heap_size / heapStats.heap_size_limit) * 100
            },
            system: {
                totalGB: Math.round(systemMem.total / 1024 / 1024 / 1024),
                freeGB: Math.round(systemMem.free / 1024 / 1024 / 1024),
                usedGB: Math.round(systemMem.used / 1024 / 1024 / 1024),
                freePercent: (systemMem.free / systemMem.total) * 100
            }
        };
    }

    checkMemoryPressure() {
        const stats = this.getMemoryStats();
        const warnings = [];

        // Check heap usage
        if (stats.heap.usagePercent > this.maxHeapWarning * 100) {
            warnings.push({
                type: 'heap_pressure',
                level: 'high',
                message: `Heap usage at ${stats.heap.usagePercent.toFixed(1)}%`,
                recommendation: 'Consider triggering GC or reducing allocations'
            });
        }

        // Check system memory
        if (stats.system.freePercent < 15) {
            warnings.push({
                type: 'system_memory',
                level: 'high',
                message: `System memory low: ${stats.system.freeGB}GB free (${stats.system.freePercent.toFixed(1)}%)`,
                recommendation: 'Close other applications or increase swap'
            });
        }

        // Check for potential string overflow conditions
        const estimatedMaxString = Math.min(
            stats.heap.limit - stats.heap.usedSize,
            this.maxStringLength
        );

        if (estimatedMaxString < 100 * 1024 * 1024) { // Less than 100MB safe space
            warnings.push({
                type: 'string_overflow_risk',
                level: 'critical',
                message: `Risk of string overflow: Only ${Math.round(estimatedMaxString / 1024 / 1024)}MB safe space`,
                recommendation: 'Trigger GC immediately or process data in chunks'
            });
        }

        return { stats, warnings };
    }

    defaultAlert(warnings, stats) {
        console.log('\n🚨 Memory Pressure Alert 🚨');
        console.log(`Time: ${new Date().toISOString()}`);
        console.log(`Heap: ${stats.heap.usagePercent.toFixed(1)}% (${Math.round(stats.heap.usedSize / 1024 / 1024)}MB / ${Math.round(stats.heap.limit / 1024 / 1024)}MB)`);
        console.log(`System: ${stats.system.usedGB}GB / ${stats.system.totalGB}GB (${stats.system.freeGB}GB free)`);
        
        warnings.forEach(warning => {
            const icon = warning.level === 'critical' ? '🔴' : '🟡';
            console.log(`${icon} ${warning.type}: ${warning.message}`);
            console.log(`   💡 ${warning.recommendation}`);
        });
        console.log('');
    }

    triggerGC() {
        if (global.gc) {
            const before = this.getMemoryStats();
            global.gc();
            const after = this.getMemoryStats();
            const freed = before.heap.usedSize - after.heap.usedSize;
            
            console.log(`🧹 Manual GC freed ${Math.round(freed / 1024 / 1024)}MB`);
            this.stats.gcCount++;
            this.stats.lastGCTime = Date.now();
            
            return freed;
        } else {
            console.log('⚠️ Manual GC not available. Start Node.js with --expose-gc flag');
            return 0;
        }
    }

    startMonitoring() {
        if (this.monitoring) return;
        
        this.monitoring = true;
        console.log('🔍 Starting M4 Pro memory monitoring...');
        console.log(`Interval: ${this.checkInterval}ms`);
        console.log(`Heap warning threshold: ${(this.maxHeapWarning * 100).toFixed(1)}%`);
        console.log(`Max safe string length: ${Math.round(this.maxStringLength / 1024 / 1024)}MB`);
        
        this.monitorInterval = setInterval(() => {
            const { stats, warnings } = this.checkMemoryPressure();
            
            // Update max heap tracking
            if (stats.heap.usedSize > this.stats.maxHeapUsed) {
                this.stats.maxHeapUsed = stats.heap.usedSize;
            }

            // Alert on warnings
            if (warnings.length > 0) {
                this.stats.pressureEvents++;
                this.alertCallback(warnings, stats);

                // Auto-trigger GC on critical warnings
                const criticalWarnings = warnings.filter(w => w.level === 'critical');
                if (criticalWarnings.length > 0 && global.gc) {
                    this.triggerGC();
                }
            }
        }, this.checkInterval);
    }

    stopMonitoring() {
        if (this.monitorInterval) {
            clearInterval(this.monitorInterval);
            this.monitoring = false;
            console.log('🛑 Memory monitoring stopped');
        }
    }

    getReport() {
        const stats = this.getMemoryStats();
        return {
            current: stats,
            session: {
                ...this.stats,
                maxHeapUsedMB: Math.round(this.stats.maxHeapUsed / 1024 / 1024),
                uptimeMinutes: Math.round(process.uptime() / 60)
            }
        };
    }

    // Utility method to safely create large strings
    static safeStringAllocation(sizeBytes, monitor) {
        if (!monitor) {
            console.warn('⚠️ No memory monitor provided for safe string allocation');
        }

        const currentStats = monitor ? monitor.getMemoryStats() : null;
        
        if (currentStats) {
            const availableHeap = currentStats.heap.limit - currentStats.heap.usedSize;
            const safeSize = Math.min(availableHeap * 0.8, sizeBytes); // Use 80% of available

            if (safeSize < sizeBytes) {
                throw new Error(`Cannot safely allocate ${Math.round(sizeBytes / 1024 / 1024)}MB string. Only ${Math.round(safeSize / 1024 / 1024)}MB available safely.`);
            }
        }

        // Proceed with allocation
        return Buffer.allocUnsafe(sizeBytes).toString();
    }
}

// CLI usage
if (require.main === module) {
    const monitor = new MemoryMonitor({
        maxHeapWarning: 0.75, // Alert at 75% heap usage
        checkInterval: 3000   // Check every 3 seconds
    });

    // Display current stats
    const initialStats = monitor.getMemoryStats();
    console.log('📊 M4 Pro Node.js Memory Status:');
    console.log(`   Heap: ${Math.round(initialStats.heap.usedSize / 1024 / 1024)}MB / ${Math.round(initialStats.heap.limit / 1024 / 1024)}MB (${initialStats.heap.usagePercent.toFixed(1)}%)`);
    console.log(`   System: ${initialStats.system.usedGB}GB / ${initialStats.system.totalGB}GB (${initialStats.system.freeGB}GB free)`);
    console.log('');

    // Start monitoring
    monitor.startMonitoring();

    // Handle graceful shutdown
    process.on('SIGINT', () => {
        console.log('\n📋 Final Memory Report:');
        const report = monitor.getReport();
        console.log(`   Session uptime: ${report.session.uptimeMinutes} minutes`);
        console.log(`   Max heap used: ${report.session.maxHeapUsedMB}MB`);
        console.log(`   Pressure events: ${report.session.pressureEvents}`);
        console.log(`   Manual GCs: ${report.session.gcCount}`);
        
        monitor.stopMonitoring();
        process.exit(0);
    });

    // Keep process alive
    process.stdin.resume();
}

module.exports = MemoryMonitor;