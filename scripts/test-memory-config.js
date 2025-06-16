#!/usr/bin/env node

/**
 * Test Node.js memory configuration and string handling
 * This script verifies that the memory optimizations work correctly
 */

console.log('🔍 Testing Node.js Memory Configuration');
console.log('=' .repeat(50));

// Test 1: Check memory settings
console.log('\n1. Memory Settings:');
const v8 = require('v8');
const heapStats = v8.getHeapStatistics();

console.log(`   Heap Size Limit: ${Math.round(heapStats.heap_size_limit / 1024 / 1024)}MB`);
console.log(`   Total Heap Size: ${Math.round(heapStats.total_heap_size / 1024 / 1024)}MB`);
console.log(`   Used Heap Size: ${Math.round(heapStats.used_heap_size / 1024 / 1024)}MB`);

// Test 2: Environment variables
console.log('\n2. Environment Variables:');
console.log(`   NODE_OPTIONS: ${process.env.NODE_OPTIONS || 'NOT SET'}`);
console.log(`   CLAUDE_CODE_MAX_OUTPUT_TOKENS: ${process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS || 'NOT SET'}`);

// Test 3: Large string allocation test
console.log('\n3. String Allocation Test:');
try {
    const testSizes = [
        { size: 10 * 1024 * 1024, name: '10MB' },
        { size: 100 * 1024 * 1024, name: '100MB' },
        { size: 500 * 1024 * 1024, name: '500MB' }
    ];
    
    for (const test of testSizes) {
        try {
            const startTime = Date.now();
            const testString = 'x'.repeat(test.size);
            const endTime = Date.now();
            
            console.log(`   ✅ ${test.name} string: ${testString.length.toLocaleString()} chars (${endTime - startTime}ms)`);
            
            // Clean up immediately
            testString.slice(0, 0);
        } catch (error) {
            console.log(`   ❌ ${test.name} string: ${error.message}`);
        }
    }
} catch (error) {
    console.log(`   ❌ String test failed: ${error.message}`);
}

// Test 4: Memory pressure test
console.log('\n4. Memory Pressure Test:');
try {
    const chunks = [];
    const chunkSize = 10 * 1024 * 1024; // 10MB chunks
    let totalAllocated = 0;
    
    for (let i = 0; i < 10; i++) {
        try {
            const chunk = Buffer.alloc(chunkSize);
            chunks.push(chunk);
            totalAllocated += chunkSize;
        } catch (error) {
            console.log(`   ⚠️  Memory pressure at ${Math.round(totalAllocated / 1024 / 1024)}MB`);
            break;
        }
    }
    
    console.log(`   ✅ Allocated ${Math.round(totalAllocated / 1024 / 1024)}MB in ${chunks.length} chunks`);
    
    // Clean up
    chunks.length = 0;
} catch (error) {
    console.log(`   ❌ Memory test failed: ${error.message}`);
}

// Test 5: Performance test
console.log('\n5. Performance Test:');
try {
    const iterations = 100000;
    const startTime = Date.now();
    
    let result = '';
    for (let i = 0; i < iterations; i++) {
        result += 'test';
    }
    
    const endTime = Date.now();
    const opsPerSec = Math.round(iterations / (endTime - startTime) * 1000);
    
    console.log(`   ✅ String concatenation: ${opsPerSec.toLocaleString()} ops/sec`);
    console.log(`   ✅ Final string length: ${result.length.toLocaleString()} chars`);
} catch (error) {
    console.log(`   ❌ Performance test failed: ${error.message}`);
}

// Test 6: Garbage collection
console.log('\n6. Garbage Collection:');
try {
    const beforeGC = v8.getHeapStatistics();
    
    if (global.gc) {
        global.gc();
        const afterGC = v8.getHeapStatistics();
        
        const freedMB = Math.round((beforeGC.used_heap_size - afterGC.used_heap_size) / 1024 / 1024);
        console.log(`   ✅ GC freed: ${freedMB}MB`);
    } else {
        console.log(`   ⚠️  GC not exposed (run with --expose-gc for full test)`);
    }
} catch (error) {
    console.log(`   ❌ GC test failed: ${error.message}`);
}

console.log('\n🏁 Memory configuration test complete!');

// Summary
const heapLimitGB = Math.round(heapStats.heap_size_limit / 1024 / 1024 / 1024);
const hasNodeOptions = process.env.NODE_OPTIONS && process.env.NODE_OPTIONS.includes('max-old-space-size');
const hasClaudeConfig = process.env.CLAUDE_CODE_MAX_OUTPUT_TOKENS;

console.log('\n📋 Summary:');
console.log(`   Heap Limit: ${heapLimitGB}GB ${heapLimitGB >= 15 ? '✅' : '❌'}`);
console.log(`   NODE_OPTIONS: ${hasNodeOptions ? '✅' : '❌'}`);
console.log(`   Claude Config: ${hasClaudeConfig ? '✅' : '❌'}`);

const allGood = heapLimitGB >= 15 && hasNodeOptions && hasClaudeConfig;
console.log(`\n${allGood ? '✅' : '❌'} Overall Status: ${allGood ? 'OPTIMIZED' : 'NEEDS CONFIGURATION'}`);

process.exit(allGood ? 0 : 1);