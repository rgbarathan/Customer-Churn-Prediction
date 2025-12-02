"""
Conversation Logger - Tracks Q&A sessions for quality assurance and training

This module logs all interactive conversations between CSRs and the AI system,
including customer info, questions, answers, confidence scores, and outcomes.
"""

import json
import os
from datetime import datetime
from pathlib import Path


class ConversationLogger:
    """Logs conversation sessions with timestamps, customer data, and metrics"""
    
    def __init__(self, logs_dir="logs"):
        """Initialize logger with logs directory"""
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(exist_ok=True)
        
        # Current session data
        self.session_data = {
            "session_id": None,
            "timestamp": None,
            "customer_id": None,
            "customer_name": None,
            "churn_probability": None,
            "csr_name": None,
            "mode": None,
            "questions": [],
            "summary": None
        }
    
    def start_session(self, customer_id, customer_name, churn_probability, 
                      csr_name="CSR", mode="interactive"):
        """Start a new conversation session"""
        self.session_data = {
            "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "customer_id": customer_id,
            "customer_name": customer_name,
            "churn_probability": float(churn_probability),
            "csr_name": csr_name,
            "mode": mode,
            "questions": [],
            "summary": None
        }
    
    def log_question(self, question, answer, confidence, category, 
                    source="unknown", is_confident=True):
        """Log a single Q&A exchange"""
        exchange = {
            "timestamp": datetime.now().isoformat(),
            "question": question,
            "answer": answer,
            "confidence": float(confidence),
            "category": category,
            "source": source,
            "is_confident": is_confident,
            "flagged_for_review": confidence < 0.5
        }
        self.session_data["questions"].append(exchange)
    
    def end_session(self, session_summary=None):
        """End the session and save log file"""
        if not self.session_data["questions"]:
            return None
        
        # Calculate session statistics
        questions_count = len(self.session_data["questions"])
        confidences = [q["confidence"] for q in self.session_data["questions"]]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        flagged_count = sum(1 for q in self.session_data["questions"] 
                           if q["flagged_for_review"])
        
        # Create summary
        self.session_data["summary"] = {
            "total_questions": questions_count,
            "average_confidence": float(avg_confidence),
            "low_confidence_count": flagged_count,
            "end_timestamp": datetime.now().isoformat(),
            "duration_seconds": (datetime.fromisoformat(self.session_data["summary"]["end_timestamp"]) 
                                - datetime.fromisoformat(self.session_data["timestamp"])).total_seconds() 
                               if self.session_data["summary"] else 0,
            "user_summary": session_summary
        }
        
        # Calculate duration properly
        start = datetime.fromisoformat(self.session_data["timestamp"])
        end = datetime.now()
        self.session_data["summary"]["duration_seconds"] = (end - start).total_seconds()
        
        # Save to file
        filename = f"{self.session_data['session_id']}.json"
        filepath = self.logs_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(self.session_data, f, indent=2)
        
        return filepath
    
    @staticmethod
    def generate_report(logs_dir="logs"):
        """Generate a report of all sessions"""
        logs_path = Path(logs_dir)
        
        if not logs_path.exists():
            return None
        
        session_files = list(logs_path.glob("session_*.json"))
        
        if not session_files:
            return None
        
        report = {
            "total_sessions": len(session_files),
            "report_generated": datetime.now().isoformat(),
            "sessions": []
        }
        
        total_questions = 0
        total_confidence = 0
        total_flagged = 0
        
        for session_file in sorted(session_files, reverse=True):
            with open(session_file, 'r') as f:
                session = json.load(f)
            
            if not session.get("summary"):
                continue
            
            session_summary = {
                "session_id": session["session_id"],
                "timestamp": session["timestamp"],
                "customer_name": session["customer_name"],
                "customer_churn_risk": f"{session['churn_probability']:.2%}",
                "csr_name": session["csr_name"],
                "questions_asked": session["summary"]["total_questions"],
                "average_confidence": f"{session['summary']['average_confidence']:.2%}",
                "low_confidence_answers": session["summary"]["low_confidence_count"],
                "duration_seconds": session["summary"]["duration_seconds"],
                "mode": session["mode"]
            }
            report["sessions"].append(session_summary)
            
            total_questions += session["summary"]["total_questions"]
            total_confidence += session["summary"]["average_confidence"]
            total_flagged += session["summary"]["low_confidence_count"]
        
        # Add aggregate statistics
        if report["sessions"]:
            report["aggregate_statistics"] = {
                "total_questions_asked": total_questions,
                "average_confidence_across_sessions": f"{total_confidence / len(report['sessions']):.2%}",
                "total_low_confidence_answers": total_flagged,
                "average_questions_per_session": f"{total_questions / len(report['sessions']):.1f}"
            }
        
        return report
    
    @staticmethod
    def save_report(report, filename="session_report.json", logs_dir="logs"):
        """Save report to file"""
        logs_path = Path(logs_dir)
        logs_path.mkdir(exist_ok=True)
        
        filepath = logs_path / filename
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        return filepath
    
    @staticmethod
    def print_report(report):
        """Print a formatted report"""
        print("\n" + "="*70)
        print("CONVERSATION SESSION REPORT")
        print("="*70)
        print(f"Report Generated: {report['report_generated']}")
        print(f"Total Sessions: {report['total_sessions']}")
        
        if "aggregate_statistics" in report:
            stats = report["aggregate_statistics"]
            print("\n📊 AGGREGATE STATISTICS:")
            print(f"  Total Questions Asked: {stats['total_questions_asked']}")
            print(f"  Average Confidence: {stats['average_confidence_across_sessions']}")
            print(f"  Low Confidence Answers: {stats['total_low_confidence_answers']}")
            print(f"  Average Questions/Session: {stats['average_questions_per_session']}")
        
        print("\n📋 SESSION DETAILS:")
        print("-" * 70)
        
        for session in report["sessions"][:10]:  # Show last 10 sessions
            print(f"\n🎯 Session ID: {session['session_id']}")
            print(f"   Timestamp: {session['timestamp']}")
            print(f"   Customer: {session['customer_name']} (Churn Risk: {session['customer_churn_risk']})")
            print(f"   CSR: {session['csr_name']}")
            print(f"   Mode: {session['mode']}")
            print(f"   Questions Asked: {session['questions_asked']}")
            print(f"   Average Confidence: {session['average_confidence']}")
            print(f"   Low Confidence Answers: {session['low_confidence_answers']}")
            print(f"   Duration: {session['duration_seconds']:.1f}s")
        
        if len(report["sessions"]) > 10:
            print(f"\n   ... and {len(report['sessions']) - 10} more sessions")
        
        print("\n" + "="*70)
    
    @staticmethod
    def export_csv(logs_dir="logs", output_file="conversations_export.csv"):
        """Export all sessions to CSV for analysis"""
        import csv
        
        logs_path = Path(logs_dir)
        
        if not logs_path.exists():
            return None
        
        session_files = list(logs_path.glob("session_*.json"))
        
        if not session_files:
            return None
        
        # Collect all Q&A pairs across all sessions
        rows = []
        
        for session_file in session_files:
            with open(session_file, 'r') as f:
                session = json.load(f)
            
            if "questions" not in session or not session["questions"]:
                continue
            
            for qa in session["questions"]:
                row = {
                    "session_id": session["session_id"],
                    "timestamp": qa["timestamp"],
                    "customer_name": session["customer_name"],
                    "customer_churn_risk": session["churn_probability"],
                    "csr_name": session["csr_name"],
                    "question": qa["question"],
                    "answer": qa["answer"],
                    "confidence": qa["confidence"],
                    "category": qa["category"],
                    "source": qa["source"],
                    "flagged_for_review": qa["flagged_for_review"]
                }
                rows.append(row)
        
        # Write to CSV
        if rows:
            output_path = Path(logs_dir) / output_file
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            
            return output_path
        
        return None


def demo_logging():
    """Demonstrate the logging system"""
    logger = ConversationLogger()
    
    # Start a sample session
    logger.start_session(
        customer_id=1,
        customer_name="John Smith",
        churn_probability=0.7963,
        csr_name="Sarah Johnson",
        mode="interactive"
    )
    
    # Log sample exchanges
    logger.log_question(
        question="How can I reduce my bill?",
        answer="Bundle discounts available for TV + Internet + Phone",
        confidence=0.6250,
        category="billing",
        source="comcast_kb"
    )
    
    logger.log_question(
        question="What loyalty programs do you have?",
        answer="service improvements",
        confidence=0.2436,
        category="retention",
        source="squad",
        is_confident=False
    )
    
    logger.log_question(
        question="Can I upgrade my internet speed?",
        answer="Yes, upgrades available in most areas",
        confidence=0.7120,
        category="services",
        source="comcast_kb"
    )
    
    # End session
    filepath = logger.end_session(
        session_summary="Customer interested in bundle upgrade"
    )
    
    print(f"✅ Session logged to: {filepath}")
    
    # Generate and display report
    report = ConversationLogger.generate_report()
    if report:
        ConversationLogger.print_report(report)
        ConversationLogger.save_report(report)
        print(f"✅ Report saved to: logs/session_report.json")
        
        # Export to CSV
        csv_path = ConversationLogger.export_csv()
        if csv_path:
            print(f"✅ CSV export saved to: {csv_path}")


if __name__ == "__main__":
    demo_logging()
