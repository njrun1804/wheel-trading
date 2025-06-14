# Jarvis 2.0 Implementation TODOs

## 🚀 Immediate Setup
- [ ] Install MLX: `pip install mlx`
- [ ] Install USearch: `pip install usearch`  
- [ ] Install PyTorch nightly: `pip install torch>=2.4.0 --pre`
- [ ] Install Metal-cpp: `brew install metal-cpp`
- [ ] Lock Metal SDK: `xcrun metal --version > .metal-sdk-version`
- [ ] Set environment variables in `.envrc`

## 📅 Week 1: Foundation

### Core Infrastructure
- [ ] Create `device_router.py` with M4 Pro detection
- [ ] Implement process manager for isolation
- [ ] Set up shared memory communication
- [ ] Configure 18GB Metal memory limit

### Storage & Search  
- [ ] Replace FAISS with USearch/hnswlib hybrid
- [ ] Migrate LMDB to DuckDB
- [ ] Create vector search benchmarks
- [ ] Test search performance on M4 Pro

### Benchmark System
- [ ] Create `bench.py` CLI tool
- [ ] Add MLX vs PyTorch comparison
- [ ] Test MCTS GPU vs CPU performance
- [ ] Document benchmark results

## 📅 Week 2: Neural & GPU

### MLX Implementation
- [ ] Port value network to MLX
- [ ] Port policy network to MLX
- [ ] Handle missing ops with CoreML
- [ ] Create PyTorch fallback system

### GPU Acceleration
- [ ] Write Metal UCB kernel for MCTS
- [ ] Implement batch node evaluation
- [ ] Benchmark GPU vs CPU tree search
- [ ] Choose optimal approach

### Integration
- [ ] Test process communication latency
- [ ] Verify GPU memory usage
- [ ] Monitor thermal behavior
- [ ] Measure end-to-end performance

## 📅 Week 3: Parallel Systems

### P-Core Search
- [ ] Create 8-worker pool for P-cores
- [ ] Implement tree merging algorithm
- [ ] Add work stealing queue
- [ ] Test parallel scaling

### E-Core Learning
- [ ] Create background worker process
- [ ] Implement DuckDB experience replay
- [ ] Add thermal-aware scheduling
- [ ] Verify non-interference

### Hardware Scheduling
- [ ] Build adaptive performance system
- [ ] Create capability detection
- [ ] Implement graceful degradation
- [ ] Document fallback paths

## 📅 Week 4: Production

### Reliability
- [ ] Add process supervision
- [ ] Implement health checks
- [ ] Create crash recovery
- [ ] Test failure scenarios

### Performance
- [ ] Profile bottlenecks
- [ ] Optimize batch sizes
- [ ] Tune memory usage
- [ ] Create monitoring dashboard

### Final Testing
- [ ] Run 24-hour thermal test
- [ ] Verify memory limits hold
- [ ] Test all fallback paths
- [ ] Create performance report

## 🎯 Quick Wins
- [ ] Add M4 Pro detection flag
- [ ] Set PyTorch MPS fallback env var
- [ ] Create memory budget config
- [ ] Build initial benchmark harness

## 📊 Success Metrics
- [ ] MCTS: 2000+ sims/sec achieved
- [ ] Neural: <50ms batch inference
- [ ] Vector: <5ms search latency
- [ ] Memory: Stays under 18GB
- [ ] Thermal: <85°C sustained

## 🔧 Daily Checklist
- [ ] Run benchmark suite
- [ ] Check thermal profile
- [ ] Monitor memory usage
- [ ] Test fallback paths
- [ ] Update performance log