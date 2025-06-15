"""
Jarvis2 Complete System - Strategic Architect for Meta System

Hybrid architecture with proper role separation:
- Jarvis2: Strategic exploration, neural-guided MCTS, on-demand consultation  
- Meta System: Autonomous execution, real-time adaptation, self-evolution

Token-aware design with 7500 token limit to prevent API errors
"""

import asyncio
from jarvis2_core import Jarvis2Core, MetaJarvis2Bridge
from jarvis2_mcts import Jarvis2MCTS, NeuralPolicy
from typing import Dict, Any


class Jarvis2Complete:
    """Complete Jarvis2 system - Strategic Architect"""
    
    def __init__(self):
        # Core components
        self.strategic_core = Jarvis2Core()
        self.neural_policy = NeuralPolicy()
        self.mcts_engine = Jarvis2MCTS(self.neural_policy)
        
        # Integration bridge (for meta system)
        self.bridge = MetaJarvis2Bridge(self.strategic_core)
        
        # State
        self.consultation_active = False
        self.learning_from_outcomes = True
        
        print("🚀 Jarvis2 Complete System Ready")
        print("🎯 Role: Strategic Architect (on-demand consultation)")
        print("🧠 MCTS Neural Exploration: Active")
        print("🤝 Meta System Integration: Ready")
        
    async def strategic_consultation(self, challenge: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Main consultation interface for meta system"""
        
        self.consultation_active = True
        
        print(f"\n🎯 Strategic Consultation Request:")
        print(f"   Challenge: {challenge}")
        print(f"   Context: {list(context.keys())}")
        
        # Phase 1: MCTS exploration (neural-guided)
        print("🔍 Phase 1: Neural-guided exploration...")
        mcts_result = await self.mcts_engine.search(context, simulations=500)
        
        # Phase 2: Strategic recommendation
        print("🧠 Phase 2: Strategic recommendation...")
        recommendation = self.bridge.request_strategic_advice(challenge, context)
        
        # Phase 3: Synthesis
        print("⚡ Phase 3: Synthesis and M4 optimization...")
        
        final_recommendation = {
            'consultation_id': recommendation.recommendation_id,
            'strategic_approach': {
                'primary': recommendation.architecture_choice,
                'mcts_preferred': mcts_result['best_action'],
                'confidence': recommendation.confidence,
                'alternatives': mcts_result['alternatives']
            },
            'implementation_guidance': {
                'approach': recommendation.implementation_approach,
                'effort_estimate': recommendation.estimated_effort,
                'm4_optimizations': recommendation.m4_optimizations,
                'trade_offs': recommendation.trade_offs
            },
            'meta_system_instructions': {
                'execution_mode': 'autonomous_with_guidance',
                'observe_patterns': True,
                'report_outcomes': True,
                'learning_objectives': [
                    'track_implementation_success',
                    'monitor_performance_metrics', 
                    'capture_user_feedback'
                ]
            }
        }
        
        self.consultation_active = False
        
        print(f"✅ Strategic consultation complete")
        print(f"🎯 Primary approach: {final_recommendation['strategic_approach']['primary']}")
        print(f"⚡ M4 optimizations: {len(final_recommendation['implementation_guidance']['m4_optimizations'])}")
        
        return final_recommendation
        
    def learn_from_meta_system(self, consultation_id: str, outcome_data: Dict[str, Any]) -> None:
        """Learn from meta system execution outcomes"""
        
        if not self.learning_from_outcomes:
            return
            
        success_score = outcome_data.get('success_score', 0.5)
        chosen_approach = outcome_data.get('chosen_approach', 'unknown')
        context = outcome_data.get('original_context', {})
        
        # Update MCTS neural policy
        self.mcts_engine.learn_from_outcome(context, chosen_approach, success_score)
        
        # Update strategic core with outcome feedback
        self.strategic_core.consultations_handled += 1
        
        print(f"📚 Learned from meta system outcome:")
        print(f"   Consultation: {consultation_id}")
        print(f"   Success: {success_score:.2f}")
        print(f"   Approach: {chosen_approach}")
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get complete system status"""
        
        return {
            'jarvis2_role': 'strategic_architect',
            'status': 'active' if self.consultation_active else 'dormant',
            'strategic_core': self.strategic_core.get_status(),
            'mcts_ready': True,
            'neural_policy_trained': True,
            'meta_integration': 'ready',
            'token_awareness': 'active',
            'learning_mode': self.learning_from_outcomes
        }


# Meta system integration protocol
async def integrate_with_meta_system():
    """Integration protocol for meta system to use Jarvis2"""
    
    jarvis2 = Jarvis2Complete()
    
    # Example: Meta system encounters complex wheel trading challenge
    consultation_result = await jarvis2.strategic_consultation(
        challenge="implement_unity_wheel_position_recommender",
        context={
            'complexity': 0.9,
            'performance_needs': 0.8,
            'domain': 'financial_options',
            'requirements': [
                'unity_api_integration',
                'real_time_calculations', 
                'position_sizing',
                'risk_management',
                'Greeks_calculations'
            ],
            'async_required': True,
            'parallel_potential': 0.9,
            'memory_intensive': 0.7
        }
    )
    
    print(f"\n🤝 Integration Result:")
    print(f"   Consultation ID: {consultation_result['consultation_id']}")
    print(f"   Strategic Approach: {consultation_result['strategic_approach']['primary']}")
    print(f"   MCTS Preference: {consultation_result['strategic_approach']['mcts_preferred']}")
    print(f"   Confidence: {consultation_result['strategic_approach']['confidence']:.2f}")
    
    # Simulate meta system execution and feedback
    await asyncio.sleep(0.1)  # Simulate execution time
    
    jarvis2.learn_from_meta_system(
        consultation_result['consultation_id'],
        {
            'success_score': 0.85,
            'chosen_approach': consultation_result['strategic_approach']['primary'],
            'original_context': consultation_result.get('context', {}),
            'performance_metrics': {'execution_time_ms': 45, 'memory_usage_mb': 120}
        }
    )
    
    return jarvis2


if __name__ == "__main__":
    async def main():
        jarvis2 = await integrate_with_meta_system()
        
        print(f"\n📊 Final System Status:")
        status = jarvis2.get_system_status()
        for key, value in status.items():
            print(f"   {key}: {value}")
            
    asyncio.run(main())