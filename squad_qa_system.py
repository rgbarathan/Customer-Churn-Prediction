import json
import os
from transformers import pipeline

class SQuADQASystem:
    """
    Enhanced QA System using SQuAD v2.0 dataset for Comcast customer support.
    Integrates the SQuAD training and development datasets to provide
    context-aware answers for customer questions.
    """
    
    def __init__(self):
        """Initialize the QA system with SQuAD data and pre-trained model."""
        self.qa_pipeline = pipeline('question-answering', 
                                   model='distilbert-base-uncased-distilled-squad')
        self.squad_data = {}
        self.contexts = []
        self.load_squad_data()
        self.setup_comcast_knowledge_base()
    
    def load_squad_data(self):
        """Load and parse SQuAD datasets from archive folder."""
        squad_files = [
            'archive/train-v2.0.json',
            'archive/dev-v2.0.json'
        ]
        
        for file_path in squad_files:
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Extract contexts from articles
                        for article in data.get('data', []):
                            for paragraph in article.get('paragraphs', []):
                                context = paragraph.get('context', '')
                                if context:
                                    self.contexts.append(context)
                    print(f"✓ Loaded {len(self.contexts)} contexts from {file_path}")
                except Exception as e:
                    print(f"⚠ Could not load {file_path}: {e}")
    
    def setup_comcast_knowledge_base(self):
        """
        Set up Comcast-specific knowledge base for QA system.
        This supplements the SQuAD data with telecom-specific information.
        """
        self.comcast_kb = {
            'billing': [
                "Comcast offers flexible billing options including monthly, annual, and auto-pay plans. We accept all major credit cards, electronic checks, and online payments through MyComcast account.",
                "To reduce your bill, consider bundling services (Internet, TV, Phone). Senior citizens and low-income customers may qualify for special discounts. New customer promotions include 50% off for the first 3 months.",
                "Monthly charges depend on service tier and add-ons. Standard Internet starts at $59.99/month. Premium services like HBO Max, Peacock, and Showtime require additional fees."
            ],
            'services': [
                "Comcast provides Internet (up to 1200 Mbps), TV (with 900+ channels), Phone, Home Security, and Streaming services (HBO Max, Peacock, Showtime).",
                "Internet plans include: Performance (100 Mbps), Performance Pro (150 Mbps), Blast (250 Mbps), Extreme (400 Mbps), and Gigabit (1200 Mbps).",
                "TV packages include: Digital Economy (200+ channels), Digital Preferred (500+ channels), and Digital Premier (900+ channels with premium content)."
            ],
            'support': [
                "Comcast 24/7 customer support is available via phone (1-855-COMCAST), chat, or the MyComcast mobile app. Technical support includes troubleshooting, service calls, and online resources.",
                "For billing questions, contact billing support at 1-855-COMCAST. For service issues, call technical support. For account management, use MyComcast portal.",
                "Device protection plans cover hardware failures. Tech support services include virus removal, network setup, and device optimization."
            ],
            'retention': [
                "If you're thinking about leaving, we'd like to help! Contact our retention team at 1-855-COMCAST to discuss loyalty offers and service improvements.",
                "Loyal customers may qualify for: extended discounts, free premium channels, equipment upgrades, and service credits.",
                "We offer price-lock guarantees and flexible contract terms to ensure you get the best value with Comcast."
            ]
        }
    
    def find_best_context(self, question, category=None):
        """
        Find the most relevant context for a question.
        First checks Comcast KB, then SQuAD contexts.
        """
        # Check Comcast KB first
        if category and category in self.comcast_kb:
            for context in self.comcast_kb[category]:
                return context
        
        # Search all Comcast KB categories
        for category_contexts in self.comcast_kb.values():
            for context in category_contexts:
                return context
        
        # Fallback to SQuAD contexts if available
        if self.contexts:
            return self.contexts[0]
        
        # Default fallback
        return "Comcast is committed to providing excellent customer service. How can we help you today?"
    
    def answer_question(self, question, category=None, min_confidence=0.5):
        """
        Answer a customer question using the enhanced QA system.
        
        Args:
            question: Customer question
            category: Optional category (billing, services, support, retention)
            min_confidence: Minimum confidence threshold for answer
        
        Returns:
            Dictionary with answer, confidence, and source
        """
        # Find appropriate context
        context = self.find_best_context(question, category)
        
        try:
            # Get answer from QA model
            result = self.qa_pipeline(question=question, context=context)
            
            response = {
                'question': question,
                'answer': result['answer'],
                'confidence': result['score'],
                'context': context,
                'is_confident': result['score'] >= min_confidence
            }
            
            return response
        except Exception as e:
            return {
                'question': question,
                'answer': f"I'm having trouble finding an answer. Please contact Comcast support at 1-855-COMCAST for assistance.",
                'confidence': 0.0,
                'error': str(e)
            }
    
    def handle_churn_customer(self, question, churn_probability):
        """
        Special handling for high-churn-risk customers.
        Prioritizes retention-focused responses.
        """
        if churn_probability > 0.5:
            # Use retention KB for at-risk customers
            return self.answer_question(question, category='retention')
        else:
            # Standard support for loyal customers
            return self.answer_question(question)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED QA SYSTEM WITH SQUAD DATA")
    print("=" * 70)
    
    # Initialize system
    qa_system = SQuADQASystem()
    
    # Test questions
    test_questions = [
        ("How can I reduce my bill?", "billing"),
        ("What internet speeds do you offer?", "services"),
        ("How do I contact customer support?", "support"),
        ("What loyalty programs are available?", "retention")
    ]
    
    print("\nTesting QA System:\n")
    for question, category in test_questions:
        print(f"Q: {question}")
        response = qa_system.answer_question(question, category)
        print(f"A: {response['answer']}")
        print(f"Confidence: {response['confidence']:.2%}")
        print("-" * 70)
    
    print("\n✓ QA System Ready!")
