"""
CSR Training System - Interactive Q&A Practice for Customer Service Representatives

This module provides guided training scenarios where CSRs can practice Q&A skills
with high-risk customers using the AI-powered Q&A system.
"""

import sys
import random
from datetime import datetime
from pathlib import Path
from squad_qa_system import SQuADQASystem
from conversation_logger import ConversationLogger


class CSRTrainingSystem:
    """Interactive training system for customer service representatives"""
    
    def __init__(self):
        """Initialize training system"""
        self.qa_system = SQuADQASystem()
        self.logger = ConversationLogger()
        
        # Training scenarios with increasing difficulty
        self.scenarios = {
            "beginner": {
                "title": "Beginner: Billing Questions",
                "description": "Practice handling basic billing inquiries",
                "difficulty": 1,
                "sample_questions": [
                    "Why is my bill so high?",
                    "Do you have any discounts available?",
                    "Can I get a bill adjustment?",
                    "What are your cheapest plans?"
                ],
                "expected_categories": ["billing"],
                "tips": [
                    "✓ Always empathize with cost concerns",
                    "✓ Explain bundling benefits",
                    "✓ Mention promotional offers",
                    "✗ Don't promise unauthorized discounts"
                ]
            },
            "intermediate": {
                "title": "Intermediate: Service & Technical Issues",
                "description": "Handle service questions and technical concerns",
                "difficulty": 2,
                "sample_questions": [
                    "What internet speeds do you offer?",
                    "How can I upgrade my service?",
                    "Why is my connection slow?",
                    "What's the difference between fiber and cable?"
                ],
                "expected_categories": ["services"],
                "tips": [
                    "✓ Explain technical features clearly",
                    "✓ Offer speed test suggestions",
                    "✓ Recommend relevant upgrades",
                    "✗ Don't make performance guarantees"
                ]
            },
            "advanced": {
                "title": "Advanced: Retention & Complex Issues",
                "description": "Retain high-risk customers with personalized solutions",
                "difficulty": 3,
                "sample_questions": [
                    "I'm thinking about switching to another provider",
                    "How can you convince me to stay?",
                    "What's your best retention offer?",
                    "I heard your competitor has better service"
                ],
                "expected_categories": ["retention"],
                "tips": [
                    "✓ Show genuine concern for customer needs",
                    "✓ Highlight unique value propositions",
                    "✓ Offer personalized retention packages",
                    "✗ Don't pressure customers to stay"
                ]
            },
            "expert": {
                "title": "Expert: Multi-Issue Resolution",
                "description": "Handle complex scenarios with multiple concerns",
                "difficulty": 4,
                "sample_questions": [
                    "I want to cancel my contract, can I leave without penalty?",
                    "Your service has been terrible and I want a refund",
                    "I found a better deal with a competitor, why should I stay?",
                    "I've been a customer for 10 years and feel unappreciated"
                ],
                "expected_categories": ["billing", "retention", "services", "support"],
                "tips": [
                    "✓ Validate customer emotions",
                    "✓ Combine multiple solutions",
                    "✓ Show company loyalty benefits",
                    "✓ Escalate if customer remains adamant",
                    "✗ Don't get defensive about complaints"
                ]
            }
        }
        
        # Training customers with various scenarios
        self.training_customers = [
            {
                "id": 101,
                "name": "Price-Sensitive Customer",
                "churn_risk": 0.65,
                "profile": "Long-time customer, unhappy with prices",
                "scenario": "beginner"
            },
            {
                "id": 102,
                "name": "Technical User",
                "churn_risk": 0.72,
                "profile": "Tech-savvy customer, wants best speed/features",
                "scenario": "intermediate"
            },
            {
                "id": 103,
                "name": "Frustrated Long-Term Customer",
                "churn_risk": 0.78,
                "profile": "10-year customer feeling undervalued",
                "scenario": "advanced"
            },
            {
                "id": 104,
                "name": "Competitor Shopping Customer",
                "churn_risk": 0.82,
                "profile": "Actively evaluating competitors",
                "scenario": "expert"
            },
            {
                "id": 105,
                "name": "New Customer - Buyer's Remorse",
                "churn_risk": 0.68,
                "profile": "Recent signup, considering cancellation",
                "scenario": "intermediate"
            }
        ]
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*70)
        print("🎓 CSR TRAINING SYSTEM - Interactive Q&A Practice")
        print("="*70)
        print("\nWelcome! This training module helps customer service representatives")
        print("practice handling high-risk customer conversations using AI-powered")
        print("answers from our knowledge base.")
        print("\n📚 Available Difficulty Levels:")
        
        for level, scenario in self.scenarios.items():
            print(f"\n  {level.upper()} (Level {scenario['difficulty']})")
            print(f"    {scenario['title']}")
            print(f"    {scenario['description']}")
    
    def select_difficulty(self):
        """Let user select training difficulty"""
        while True:
            print("\n" + "-"*70)
            print("Select difficulty level:")
            print("  1) Beginner - Billing Questions")
            print("  2) Intermediate - Service & Technical Issues")
            print("  3) Advanced - Retention & Complex Issues")
            print("  4) Expert - Multi-Issue Resolution")
            print("  5) Random Customer (any level)")
            print("  6) Return to main menu")
            
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == "1":
                return "beginner"
            elif choice == "2":
                return "intermediate"
            elif choice == "3":
                return "advanced"
            elif choice == "4":
                return "expert"
            elif choice == "5":
                return random.choice(list(self.scenarios.keys()))
            elif choice == "6":
                return None
            else:
                print("❌ Invalid choice. Please enter 1-6.")
    
    def select_customer(self, difficulty=None):
        """Select or random customer for training"""
        if difficulty:
            candidates = [c for c in self.training_customers 
                         if c["scenario"] == difficulty]
            if candidates:
                return random.choice(candidates)
        
        return random.choice(self.training_customers)
    
    def run_training_session(self, customer, difficulty):
        """Run a training session with a customer"""
        print("\n" + "="*70)
        print(f"🎯 TRAINING SESSION - {self.scenarios[difficulty]['title']}")
        print("="*70)
        
        # Show customer profile
        print(f"\n👤 Customer Profile:")
        print(f"   Name: {customer['name']}")
        print(f"   ID: {customer['id']}")
        print(f"   Churn Risk: {customer['churn_risk']:.1%}")
        print(f"   Profile: {customer['profile']}")
        print(f"   Scenario Type: {difficulty.upper()}")
        
        # Show tips
        print(f"\n💡 Tips for this scenario:")
        for tip in self.scenarios[difficulty]["tips"]:
            print(f"   {tip}")
        
        # Start training session
        print(f"\n📖 SAMPLE QUESTIONS (for reference):")
        for i, q in enumerate(self.scenarios[difficulty]["sample_questions"], 1):
            print(f"   {i}. {q}")
        
        # Initialize logger
        self.logger.start_session(
            customer_id=customer['id'],
            customer_name=customer['name'],
            churn_probability=customer['churn_risk'],
            csr_name="Training CSR",
            mode="training"
        )
        
        # Interactive training loop
        print(f"\n🎤 Your Practice Session")
        print("-"*70)
        print("💬 Pretend to be a customer and ask questions OR")
        print("   Type 'ask' to get a sample question")
        print("   Type 'hint' for AI suggestions")
        print("   Type 'end' to finish training")
        print("-"*70)
        
        question_count = 0
        total_confidence = 0
        
        while True:
            user_input = input("\n🎤 Your question (or command): ").strip()
            
            if user_input.lower() in ['end', 'quit', 'exit', 'q']:
                break
            
            if user_input.lower() == 'ask':
                # Show sample question
                question = random.choice(
                    self.scenarios[difficulty]["sample_questions"]
                )
                print(f"\n💭 Sample customer question: {question}")
                print("   (Try responding to this)")
                continue
            
            if user_input.lower() == 'hint':
                print(f"\n💡 Remember these tips:")
                for tip in self.scenarios[difficulty]["tips"]:
                    print(f"   {tip}")
                continue
            
            # Get AI answer
            print(f"\n🤔 Searching knowledge base...")
            qa_response = self.qa_system.answer_question(
                user_input,
                category=difficulty if difficulty in ["billing", "retention", "services"] 
                         else "support"
            )
            
            answer = qa_response["answer"]
            confidence = qa_response["confidence"]
            
            # Display AI response
            print(f"\n🤖 AI Suggested Answer:")
            print(f"   {answer}")
            print(f"   Confidence: {confidence:.2%}")
            
            # Provide feedback based on confidence
            if confidence > 0.70:
                print(f"   ✅ HIGH CONFIDENCE - Safe to use this answer")
                feedback = "high"
            elif confidence > 0.50:
                print(f"   ⚠️  MEDIUM CONFIDENCE - Good, but verify with supervisor if unsure")
                feedback = "medium"
            else:
                print(f"   ❌ LOW CONFIDENCE - Get supervisor approval before using")
                feedback = "low"
            
            # Ask CSR to formulate response
            csr_response = input("\n📝 How would you explain this to the customer? ")
            
            if csr_response.strip():
                # Score the response
                score = self._score_response(
                    csr_response,
                    answer,
                    confidence,
                    difficulty
                )
                
                print(f"\n📊 Response Score: {score}/10")
                
                # Log the interaction
                self.logger.log_question(
                    question=user_input,
                    answer=answer,
                    confidence=confidence,
                    category=difficulty if difficulty in ["billing", "retention", "services"] 
                             else "support",
                    source="training",
                    is_confident=(confidence > 0.50)
                )
                
                question_count += 1
                total_confidence += confidence
            
            # Ask if they want to try another question
            continue_training = input("\n❓ Ask another question? (yes/no): ").strip().lower()
            if continue_training in ['no', 'n', 'exit', 'quit']:
                break
        
        # End session and show results
        self.logger.end_session(session_summary=f"Training mode: {difficulty}")
        
        print(f"\n" + "="*70)
        print(f"📋 TRAINING SESSION RESULTS")
        print(f"="*70)
        print(f"Questions Asked: {question_count}")
        if question_count > 0:
            avg_confidence = total_confidence / question_count
            print(f"Average Confidence: {avg_confidence:.2%}")
            print(f"Scenario: {self.scenarios[difficulty]['title']}")
            
            if avg_confidence > 0.70:
                print(f"✅ Excellent! You handled high-confidence answers well.")
            elif avg_confidence > 0.50:
                print(f"👍 Good job! Remember to verify low-confidence answers.")
            else:
                print(f"💪 Keep practicing! Work on selecting better responses.")
        
        print(f"="*70)
    
    def _score_response(self, csr_response, ai_answer, confidence, difficulty):
        """Score CSR's response quality (0-10)"""
        score = 5  # Base score
        
        # Bonus for acknowledging AI answer
        if any(word in csr_response.lower() for word in 
               ["available", "offer", "have", "yes", "can", "bundle", "discount"]):
            score += 2
        
        # Bonus for personalization
        if any(word in csr_response.lower() for word in 
               ["you", "your", "customer", "help", "recommend", "personal"]):
            score += 1
        
        # Bonus for length (more detailed = better)
        if len(csr_response) > 30:
            score += 1
        
        # Bonus for high confidence answers
        if confidence > 0.70:
            score += 1
        
        # Penalty for missing key points
        if difficulty == "beginner" and "discount" not in csr_response.lower():
            score -= 1
        
        if difficulty == "advanced" and "loyalty" not in csr_response.lower():
            score -= 1
        
        return min(10, max(0, score))
    
    def view_training_report(self):
        """View all training sessions"""
        report = ConversationLogger.generate_report()
        
        if not report or report["total_sessions"] == 0:
            print("\n❌ No training sessions found.")
            return
        
        print("\n" + "="*70)
        print("📊 TRAINING REPORT")
        print("="*70)
        
        # Filter for training sessions
        training_sessions = [s for s in report.get("sessions", []) 
                            if s.get("mode") == "training"]
        
        if not training_sessions:
            print("❌ No training sessions in log.")
            return
        
        print(f"\nTotal Training Sessions: {len(training_sessions)}")
        
        total_q = sum(s["questions_asked"] for s in training_sessions)
        avg_conf = sum(float(s["average_confidence"].strip('%')) 
                      for s in training_sessions) / len(training_sessions) / 100
        
        print(f"Total Questions Asked: {total_q}")
        print(f"Average Confidence: {avg_conf:.2%}")
        
        print(f"\n📋 Recent Training Sessions:")
        print("-"*70)
        
        for session in training_sessions[:5]:
            print(f"\n🎯 {session['session_id']}")
            print(f"   Customer: {session['customer_name']} ({session['customer_churn_risk']})")
            print(f"   Questions: {session['questions_asked']}")
            print(f"   Avg Confidence: {session['average_confidence']}")
            print(f"   Timestamp: {session['timestamp']}")
        
        print(f"\n" + "="*70)
    
    def main_menu(self):
        """Main training menu"""
        self.display_welcome()
        
        while True:
            print("\n" + "="*70)
            print("MAIN MENU")
            print("="*70)
            print("1) Start Training Session")
            print("2) View Training Report")
            print("3) View Sample Scenarios")
            print("4) Exit Training")
            
            choice = input("\nEnter your choice (1-4): ").strip()
            
            if choice == "1":
                difficulty = self.select_difficulty()
                if difficulty:
                    customer = self.select_customer(difficulty)
                    self.run_training_session(customer, difficulty)
            
            elif choice == "2":
                self.view_training_report()
            
            elif choice == "3":
                self.show_scenarios()
            
            elif choice == "4":
                print("\n✅ Training complete! Keep practicing to improve your skills.")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-4.")
    
    def show_scenarios(self):
        """Show all available scenarios"""
        print("\n" + "="*70)
        print("📚 AVAILABLE TRAINING SCENARIOS")
        print("="*70)
        
        for level, scenario in self.scenarios.items():
            print(f"\n🎯 {scenario['title']} (Difficulty: {scenario['difficulty']}/4)")
            print(f"   {scenario['description']}")
            print(f"   Sample Questions:")
            for q in scenario['sample_questions'][:2]:
                print(f"     • {q}")
            print(f"   Tips:")
            for tip in scenario['tips'][:2]:
                print(f"     {tip}")


def main():
    """Run training system"""
    training = CSRTrainingSystem()
    training.main_menu()


if __name__ == "__main__":
    main()
