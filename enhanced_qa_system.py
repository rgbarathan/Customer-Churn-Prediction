"""
Enhanced QA System with Customer-Aware Responses

This implementation addresses real business needs:
1. Customer-specific context based on their profile
2. Risk-based response templates
3. Better context matching with keyword scoring
4. Actionable recommendations tied to churn factors
"""

import json
import os
from transformers import pipeline
import re

class EnhancedQASystem:
    """
    Customer-aware QA System that generates personalized retention strategies
    based on churn risk factors and customer profile.
    """
    
    def __init__(self):
        """Initialize QA system with pre-trained model and enhanced knowledge base."""
        self.qa_pipeline = pipeline('question-answering', 
                                   model='distilbert-base-uncased-distilled-squad')
        self.setup_enhanced_knowledge_base()
        print("✓ Loaded DistilBERT model (pre-trained on SQuAD v1.1)")
        print("✓ Initialized enhanced customer-aware knowledge base")
    
    def setup_enhanced_knowledge_base(self):
        """
        Enhanced knowledge base with customer profile-specific contexts.
        Organized by risk factors identified in churn prediction.
        """
        self.knowledge_base = {
            # Billing-related contexts
            'high_charges': {
                'context': "For customers with high monthly charges, we offer several cost-reduction options. Bundle discounts can save 20-30% by combining Internet, TV, and Phone services. Our loyalty discount program provides up to $25/month off for customers who have been with us over 12 months. Senior citizens (65+) qualify for an additional $15/month discount. New customers receive 50% off their first 3 months. We also offer flexible payment plans to spread costs over time without penalties.",
                'keywords': ['bill', 'cost', 'expensive', 'high', 'price', 'reduce', 'lower', 'save', 'discount', 'cheaper']
            },
            'senior_discount': {
                'context': "Senior citizens aged 65 and older qualify for our Senior Advantage Program. This includes: $15/month discount on any Internet plan, free tech support and installation, priority customer service, discounted equipment rentals at $5/month instead of $12/month, and no early termination fees. Seniors can also bundle Internet with our LifeLine phone service for an additional $10/month savings. To enroll, simply provide proof of age (driver's license or ID) to any Comcast representative.",
                'keywords': ['senior', 'elderly', 'age', 'retirement', 'older', '65']
            },
            'bundle_savings': {
                'context': "Bundling multiple services provides significant savings and increased convenience. Our Triple Play bundle (Internet + TV + Phone) starts at $99/month, saving customers up to $50/month compared to individual services. Double Play options (any two services) save $30/month. Bundled customers also receive: free installation, one bill for all services, premium channels at no extra cost for 6 months, and enhanced customer support. Bundles also increase service stickiness and reduce the likelihood of switching providers.",
                'keywords': ['bundle', 'package', 'combine', 'together', 'triple', 'double', 'multiple services']
            },
            
            # Service-related contexts
            'internet_plans': {
                'context': "Comcast Internet plans are designed for different usage needs and budgets. Performance (100 Mbps) at $59.99/month is ideal for light browsing and streaming. Performance Pro (150 Mbps) at $69.99/month supports multiple devices. Blast (250 Mbps) at $79.99/month is perfect for gaming and 4K streaming. Extreme (400 Mbps) at $89.99/month handles smart homes with many connected devices. Gigabit (1200 Mbps) at $99.99/month is our fastest option for power users and home offices. All plans include free modem rental for the first year.",
                'keywords': ['internet', 'speed', 'mbps', 'fast', 'slow', 'connection', 'wifi', 'broadband']
            },
            'service_upgrades': {
                'context': "Upgrading your service is easy and often comes with promotional pricing. Current customers can upgrade to higher-speed Internet for just $10-20/month more. Add TV service to your Internet plan for only $40/month (normally $70). Premium channels like HBO Max, Showtime, and Peacock can be added for $10-15 each. We also offer free equipment upgrades when you commit to a 12-month contract. Tech support packages with 24/7 assistance are available for $15/month. Most upgrades take effect within 24 hours with no service interruption.",
                'keywords': ['upgrade', 'better', 'faster', 'more', 'add', 'improve', 'enhance']
            },
            'fiber_vs_dsl': {
                'context': "Fiber optic Internet provides superior speed and reliability compared to DSL. Fiber offers symmetrical upload and download speeds (1000 Mbps both ways), while DSL typically maxes out at 100 Mbps download and 10 Mbps upload. Fiber has lower latency (under 10ms vs 20-50ms for DSL), making it better for gaming and video calls. Fiber is more reliable during peak hours and weather conditions. We offer fiber-to-the-home installation for $99 one-time fee, and fiber plans start at $79.99/month. Customers switching from DSL to Fiber report 95% satisfaction rates.",
                'keywords': ['fiber', 'dsl', 'cable', 'connection type', 'technology', 'difference']
            },
            
            # Contract and commitment contexts
            'contract_flexibility': {
                'context': "We offer flexible contract options to match your needs. Month-to-month contracts provide maximum flexibility with no commitment or early termination fees. 12-month contracts save you $15/month and include free equipment. 24-month contracts offer the best value with $25/month savings, free premium channels for 6 months, and free professional installation worth $100. All contracts include our 30-day money-back guarantee. You can upgrade services anytime without penalty, and we offer contract pause options for extended travel or temporary situations.",
                'keywords': ['contract', 'commitment', 'term', 'agreement', 'cancel', 'leave', 'switch', 'terminate']
            },
            'early_cancellation': {
                'context': "If you're considering cancellation, please let us help first. We have dedicated retention specialists who can offer exclusive deals not available to new customers. Early termination fees range from $10-20 per month remaining on contract, but we can often waive these fees if there's a service issue or financial hardship. Customers who cancel often regret it when they see competitor pricing and service quality. Before canceling, let us review your account - 80% of customers who call to cancel end up staying after speaking with our retention team.",
                'keywords': ['cancel', 'quit', 'leave', 'terminate', 'stop', 'end service', 'disconnect']
            },
            
            # Retention and loyalty contexts  
            'loyalty_rewards': {
                'context': "Our Customer Loyalty Program rewards long-term customers with exclusive benefits. After 12 months: $10/month discount and priority support. After 24 months: $20/month discount, free equipment upgrades, and no service fees. After 36+ months: $30/month discount, dedicated account manager, free premium channels, and special pricing on new services. Loyal customers also get first access to new technologies like WiFi 6 routers and streaming devices. We also offer referral bonuses - get $50 credit for each friend who signs up. Loyalty discounts are automatically applied and increase over time.",
                'keywords': ['loyalty', 'reward', 'long-term', 'tenure', 'stay', 'keep', 'remain', 'continue']
            },
            'price_lock': {
                'context': "Our Price Lock Guarantee protects you from unexpected rate increases. With a 24-month contract, your rate is locked for the full term - no surprise price hikes. After the initial term, we offer a Price Protection Plan that limits increases to no more than 3% per year, compared to 5-10% typical industry increases. Price Lock includes all base services but excludes premium add-ons you choose to add later. This guarantee has saved our customers an average of $300 over two years compared to month-to-month pricing. Long-term customers on Price Lock have 40% lower churn rates.",
                'keywords': ['price lock', 'guarantee', 'rate', 'increase', 'hike', 'cost', 'stable']
            },
            'competitive_match': {
                'context': "We will match or beat competitor offers for qualified customers. If you have a written quote from another provider (AT&T, Verizon, Spectrum), bring it to us within 7 days and we'll match the price or beat it by $10/month. Our price match includes equivalent or better service levels - we won't downgrade your speed or features. This offer applies to both new and existing customers. Additionally, we'll cover any cancellation fees from your previous provider up to $500 with proof of charges. 90% of customers who bring us competitor quotes choose to stay with Comcast after seeing our match offer.",
                'keywords': ['competitor', 'other provider', 'at&t', 'verizon', 'spectrum', 'cheaper elsewhere', 'match']
            },
            
            # Support and service quality contexts
            'tech_support': {
                'context': "Comcast provides comprehensive technical support to ensure your service runs smoothly. Basic support is available 24/7 via phone (1-855-COMCAST), live chat, and the MyComcast app at no charge. Premium Tech Support ($15/month) includes: unlimited on-site visits, same-day appointments, whole-home WiFi setup, virus and malware removal, device setup and optimization, and dedicated support line with under 2-minute wait times. We also offer remote support where technicians can access your equipment to diagnose and fix issues without a home visit. Average issue resolution time is under 30 minutes.",
                'keywords': ['support', 'help', 'technical', 'problem', 'issue', 'broken', 'not working', 'fix']
            },
            'service_quality': {
                'context': "Comcast has invested over $10 billion in network infrastructure to provide reliable, high-quality service. Our network uptime is 99.9%, meaning less than 9 hours of downtime per year. We guarantee minimum speeds of 80% of advertised rates during peak hours, and typically deliver 100-120% of advertised speeds. Service quality issues are rare but taken seriously - we offer automatic credits for extended outages and priority repair for frequent issues. Our customer satisfaction scores have improved 25% over the past 2 years. We also provide real-time network status updates through the MyComcast app.",
                'keywords': ['quality', 'reliable', 'uptime', 'outage', 'down', 'slow', 'performance']
            },
            
            # New customer contexts
            'new_customer_offers': {
                'context': "New customers receive our best promotional pricing. Current offers include: 50% off all plans for the first 3 months, free professional installation (normally $100), free premium modem/router (value $200), and 3 months of premium channels at no cost. After the promotional period, prices adjust to standard rates, but new customers can lock in their rate with a 24-month contract. We also offer a 30-day money-back guarantee - if you're not satisfied, cancel within 30 days for a full refund with no questions asked. These promotions are only available to customers who haven't had service in the past 90 days.",
                'keywords': ['new customer', 'new', 'sign up', 'join', 'start', 'first time']
            },
            'onboarding_support': {
                'context': "New customers receive dedicated onboarding support for the first 60 days. This includes: welcome call from your account manager within 48 hours of activation, free in-home setup assistance (up to 2 hours), tutorial on using MyComcast app and online account management, and priority technical support with no wait times. We'll help you set up all your devices, optimize your WiFi coverage, and ensure you understand your billing. During the first 60 days, you have access to our New Customer Success team who can answer any questions. Studies show customers who use onboarding support have 60% lower early-stage churn.",
                'keywords': ['new', 'setup', 'start', 'begin', 'onboarding', 'getting started', 'first time']
            }
        }
    
    def find_best_context(self, question, customer_profile=None):
        """
        Intelligently match question to best context using keyword scoring
        and customer profile analysis.
        
        Args:
            question: Customer's question
            customer_profile: Dict with customer attributes (tenure, charges, services, etc.)
        
        Returns:
            Best matching context string
        """
        question_lower = question.lower()
        scores = {}
        
        # Score each context based on keyword matches
        for context_name, context_data in self.knowledge_base.items():
            score = 0
            for keyword in context_data['keywords']:
                if keyword in question_lower:
                    score += 1
            scores[context_name] = score
        
        # Boost scores based on customer profile if provided
        if customer_profile:
            # High charges → prioritize billing contexts
            if customer_profile.get('monthly_charges', 0) > 90:
                scores['high_charges'] = scores.get('high_charges', 0) + 2
                scores['bundle_savings'] = scores.get('bundle_savings', 0) + 1
            
            # Senior citizen → prioritize senior discount
            if customer_profile.get('senior_citizen'):
                scores['senior_discount'] = scores.get('senior_discount', 0) + 3
            
            # Short tenure → prioritize new customer and loyalty
            if customer_profile.get('tenure', 0) < 6:
                scores['new_customer_offers'] = scores.get('new_customer_offers', 0) + 2
                scores['onboarding_support'] = scores.get('onboarding_support', 0) + 1
            
            # Long tenure → prioritize loyalty rewards
            if customer_profile.get('tenure', 0) > 24:
                scores['loyalty_rewards'] = scores.get('loyalty_rewards', 0) + 3
            
            # Month-to-month contract → prioritize contract benefits
            if customer_profile.get('contract') == 'month-to-month':
                scores['contract_flexibility'] = scores.get('contract_flexibility', 0) + 2
            
            # Minimal services → suggest upgrades and bundles
            if customer_profile.get('service_count', 0) <= 1:
                scores['bundle_savings'] = scores.get('bundle_savings', 0) + 2
                scores['service_upgrades'] = scores.get('service_upgrades', 0) + 1
        
        # Find context with highest score
        if scores:
            best_context_name = max(scores, key=scores.get)
            if scores[best_context_name] > 0:
                return self.knowledge_base[best_context_name]['context']
        
        # Fallback to first relevant context
        for context_name, context_data in self.knowledge_base.items():
            if any(kw in question_lower for kw in context_data['keywords']):
                return context_data['context']
        
        # Ultimate fallback
        return "Comcast is committed to providing excellent customer service and competitive pricing. Let me connect you with a retention specialist who can review your specific account and provide personalized offers. Please call 1-855-COMCAST and mention you're a valued customer considering your options."
    
    def answer_question(self, question, customer_profile=None, min_confidence=0.3):
        """
        Answer customer question with profile-aware context selection.
        
        Args:
            question: Customer's question
            customer_profile: Dict with customer data (tenure, charges, senior, etc.)
            min_confidence: Minimum confidence threshold
        
        Returns:
            Dict with answer, confidence, context, and recommendations
        """
        # Find best context based on question and customer profile
        context = self.find_best_context(question, customer_profile)
        
        try:
            # Get answer from QA model
            result = self.qa_pipeline(question=question, context=context)
            
            # Generate personalized recommendation based on customer profile
            recommendations = self._generate_recommendations(customer_profile)
            
            response = {
                'question': question,
                'answer': result['answer'],
                'confidence': result['score'],
                'context': context,
                'is_confident': result['score'] >= min_confidence,
                'recommendations': recommendations,
                'customer_specific': customer_profile is not None
            }
            
            return response
        except Exception as e:
            return {
                'question': question,
                'answer': "I apologize, I'm having trouble processing that question. Let me connect you with a specialist at 1-855-COMCAST.",
                'confidence': 0.0,
                'error': str(e),
                'recommendations': self._generate_recommendations(customer_profile)
            }
    
    def _generate_recommendations(self, customer_profile):
        """
        Generate actionable recommendations based on customer profile.
        
        This provides specific actions CSR should take based on churn risk factors.
        """
        if not customer_profile:
            return ["Contact customer with personalized retention offer"]
        
        recommendations = []
        
        # High charges
        if customer_profile.get('monthly_charges', 0) > 90:
            recommendations.append("⚡ Offer bundle discount to reduce monthly cost")
            if customer_profile.get('service_count', 0) <= 1:
                recommendations.append("📦 Suggest Triple Play bundle - can save $50/month")
        
        # Senior citizen
        if customer_profile.get('senior_citizen'):
            recommendations.append("👴 Apply Senior Advantage Program ($15/month discount)")
            recommendations.append("💡 Mention free tech support for seniors")
        
        # Short tenure (high churn risk)
        if customer_profile.get('tenure', 0) < 6:
            recommendations.append("🆕 Buyer's remorse period - offer new customer retention deal")
            recommendations.append("📞 Schedule follow-up call in 2 weeks to ensure satisfaction")
            if customer_profile.get('tenure', 0) <= 2:
                recommendations.append("🚨 URGENT: First 90 days critical - offer 50% off next 3 months")
        
        # Long tenure (reward loyalty)
        if customer_profile.get('tenure', 0) > 24:
            recommendations.append("⭐ Loyal customer - offer loyalty rewards ($20-30/month discount)")
            recommendations.append("🎁 Free equipment upgrade to show appreciation")
        
        # Month-to-month contract (easy to leave)
        if customer_profile.get('contract') == 'month-to-month':
            recommendations.append("📝 Convert to 12 or 24-month contract with discounted rate")
            recommendations.append("🔒 Offer Price Lock Guarantee to prevent rate increases")
        
        # Minimal services (low engagement)
        if customer_profile.get('service_count', 0) <= 1:
            recommendations.append("📺 Low service engagement - add free premium channels (6 months)")
            recommendations.append("🎮 Suggest service add-ons with intro pricing")
        
        # No add-ons (lower switching cost)
        if not customer_profile.get('has_addons'):
            recommendations.append("🛡️ Add free security/backup services to increase stickiness")
            recommendations.append("💼 Offer free tech support trial (3 months)")
        
        # Fiber service (premium expectations)
        if customer_profile.get('internet_type') == 'fiber':
            recommendations.append("🚀 Premium customer - ensure service quality is excellent")
            recommendations.append("💎 Offer dedicated account manager")
        
        # Default fallback
        if not recommendations:
            recommendations.append("📋 Review account for personalized retention offer")
            recommendations.append("🤝 Connect with retention specialist")
        
        return recommendations
    
    def handle_churn_customer(self, question, churn_probability, customer_profile=None):
        """
        Special handling for high-churn-risk customers with enhanced personalization.
        
        Args:
            question: Customer's question
            churn_probability: Predicted churn probability (0-1)
            customer_profile: Customer attributes dict
        
        Returns:
            Enhanced response with risk-based recommendations
        """
        # Get base answer
        response = self.answer_question(question, customer_profile)
        
        # Add churn risk context
        response['churn_probability'] = churn_probability
        response['risk_level'] = self._get_risk_level(churn_probability)
        
        # Add urgency-based messaging
        if churn_probability > 0.7:
            response['urgency'] = 'CRITICAL'
            response['action_timeline'] = 'Contact within 24 hours'
            response['escalation'] = 'Escalate to senior retention specialist'
        elif churn_probability > 0.5:
            response['urgency'] = 'HIGH'
            response['action_timeline'] = 'Contact within 48 hours'
            response['escalation'] = 'Retention team handle'
        elif churn_probability > 0.3:
            response['urgency'] = 'MEDIUM'
            response['action_timeline'] = 'Monitor and follow up within 1 week'
            response['escalation'] = 'Standard retention process'
        else:
            response['urgency'] = 'LOW'
            response['action_timeline'] = 'Routine check-in'
            response['escalation'] = 'None required'
        
        return response
    
    def _get_risk_level(self, probability):
        """Convert probability to risk level label."""
        if probability > 0.7:
            return 'CRITICAL'
        elif probability > 0.5:
            return 'HIGH'
        elif probability > 0.3:
            return 'MEDIUM'
        else:
            return 'LOW'


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("ENHANCED CUSTOMER-AWARE QA SYSTEM")
    print("=" * 70)
    
    # Initialize system
    qa_system = EnhancedQASystem()
    
    # Test case 1: Senior citizen with high charges
    print("\n" + "=" * 70)
    print("TEST 1: Senior Citizen with High Charges")
    print("=" * 70)
    customer1 = {
        'senior_citizen': True,
        'monthly_charges': 105,
        'tenure': 2,
        'service_count': 1,
        'contract': 'month-to-month',
        'has_addons': False
    }
    
    result = qa_system.handle_churn_customer(
        question="How can I reduce my bill?",
        churn_probability=0.65,
        customer_profile=customer1
    )
    
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Urgency: {result['urgency']}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")
    
    # Test case 2: Long-term loyal customer
    print("\n" + "=" * 70)
    print("TEST 2: Long-term Loyal Customer")
    print("=" * 70)
    customer2 = {
        'senior_citizen': False,
        'monthly_charges': 70,
        'tenure': 29,
        'service_count': 5,
        'contract': 'two-year',
        'has_addons': True
    }
    
    result = qa_system.handle_churn_customer(
        question="What loyalty programs are available?",
        churn_probability=0.05,
        customer_profile=customer2
    )
    
    print(f"Q: {result['question']}")
    print(f"A: {result['answer']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"Risk Level: {result['risk_level']}")
    print(f"Urgency: {result['urgency']}")
    print(f"\nRecommendations:")
    for rec in result['recommendations']:
        print(f"  {rec}")
    
    print("\n" + "=" * 70)
    print("✓ Enhanced QA System Ready!")
    print("=" * 70)
